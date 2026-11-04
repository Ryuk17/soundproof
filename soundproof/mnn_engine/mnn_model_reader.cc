#include "mnn_model_reader.h"

#include <MNN/MNNDefine.h>

#include <chrono>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace soundproof {
namespace mnn_engine {
namespace {

constexpr int kMaxAttempts = 3;
constexpr int kRetryDelayMs = 20;

#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& s) {
  if (s.empty()) {
    return std::wstring();
  }
  const int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  if (len <= 1) {
    return std::wstring();
  }
  std::wstring wide(static_cast<size_t>(len), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &wide[0], len);
  wide.resize(static_cast<size_t>(len - 1));  // drop the terminating null
  return wide;
}

// Returns true on success. A failed attempt may be caused by a concurrent
// writer, so the caller retries.
bool ReadOnce(const std::wstring& path, std::vector<uint8_t>* out) {
  HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    MNN_ERROR("[mnn_engine] cannot open model file (error %lu)\n",
              static_cast<unsigned long>(GetLastError()));
    return false;
  }

  // Windows has no shared file lock; an exclusive lock serializes concurrent
  // loads across processes. Failure is not fatal: some file systems do not
  // support byte-range locks.
  OVERLAPPED overlapped{};
  const bool locked =
      LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &overlapped) != 0;
  if (!locked) {
    MNN_PRINT("[mnn_engine] model file lock unavailable, continue unlocked\n");
  }

  bool ok = false;
  LARGE_INTEGER size{};
  do {
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0) {
      MNN_ERROR("[mnn_engine] model file is empty or its size is unavailable\n");
      break;
    }
    out->resize(static_cast<size_t>(size.QuadPart));

    size_t total = 0;
    while (total < out->size()) {
      const size_t remaining = out->size() - total;
      const DWORD to_read = static_cast<DWORD>(
          remaining > (1u << 30) ? (1u << 30) : remaining);
      DWORD read = 0;
      if (!ReadFile(file, out->data() + total, to_read, &read, nullptr) ||
          read == 0) {
        MNN_ERROR("[mnn_engine] reading model file failed\n");
        break;
      }
      total += read;
    }
    if (total != out->size()) {
      break;
    }

    // Detect a writer that ignored the lock and truncated/grew the file.
    LARGE_INTEGER after{};
    if (!GetFileSizeEx(file, &after) || after.QuadPart != size.QuadPart) {
      MNN_ERROR("[mnn_engine] model file changed while being read, retrying\n");
      break;
    }
    ok = true;
  } while (false);

  if (locked) {
    UnlockFileEx(file, 0, 1, 0, &overlapped);
  }
  CloseHandle(file);
  return ok;
}
#else
bool ReadOnce(const std::string& path, std::vector<uint8_t>* out) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    MNN_ERROR("[mnn_engine] cannot open model file '%s' (errno %d)\n",
              path.c_str(), errno);
    return false;
  }

  const bool locked = ::flock(fd, LOCK_SH) == 0;
  if (!locked) {
    MNN_PRINT("[mnn_engine] model file lock unavailable, continue unlocked\n");
  }

  bool ok = false;
  struct stat before {};
  do {
    if (::fstat(fd, &before) != 0 || before.st_size <= 0) {
      MNN_ERROR("[mnn_engine] model file '%s' is empty or unreadable\n",
                path.c_str());
      break;
    }
    out->resize(static_cast<size_t>(before.st_size));

    size_t total = 0;
    while (total < out->size()) {
      const ssize_t n = ::read(fd, out->data() + total, out->size() - total);
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        MNN_ERROR("[mnn_engine] reading model file '%s' failed (errno %d)\n",
                  path.c_str(), errno);
        break;
      }
      if (n == 0) {
        MNN_ERROR("[mnn_engine] unexpected end of model file '%s'\n",
                  path.c_str());
        break;
      }
      total += static_cast<size_t>(n);
    }
    if (total != out->size()) {
      break;
    }

    // Detect a writer that ignored the lock. Atomic rename is safe: our fd
    // still refers to the old inode, whose size/mtime do not change.
    struct stat after {};
    if (::fstat(fd, &after) != 0 || after.st_size != before.st_size ||
        after.st_mtime != before.st_mtime) {
      MNN_ERROR("[mnn_engine] model file '%s' changed while being read\n",
                path.c_str());
      break;
    }
    ok = true;
  } while (false);

  if (locked) {
    ::flock(fd, LOCK_UN);
  }
  ::close(fd);
  return ok;
}
#endif  // _WIN32

}  // namespace

bool ReadModelFile(const std::string& path, std::vector<uint8_t>* out) {
  if (out == nullptr || path.empty()) {
    MNN_ERROR("[mnn_engine] ReadModelFile: invalid argument\n");
    return false;
  }
  out->clear();
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
#ifdef _WIN32
    if (ReadOnce(Utf8ToWide(path), out)) {
#else
    if (ReadOnce(path, out)) {
#endif
      return true;
    }
    if (attempt + 1 < kMaxAttempts) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs));
    }
  }
  MNN_ERROR("[mnn_engine] failed to read model file '%s' after %d attempts\n",
            path.c_str(), kMaxAttempts);
  out->clear();
  return false;
}

}  // namespace mnn_engine
}  // namespace soundproof
