#ifndef SOUNDPROOF_CORE_MNN_ENGINE_MNN_MODEL_READER_H_
#define SOUNDPROOF_CORE_MNN_ENGINE_MNN_MODEL_READER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace soundproof {
namespace mnn_engine {

// Reads a whole model file into |out|.
//
// Multi-process safety: while reading, the file is locked with an OS advisory
// lock (shared flock on POSIX, exclusive LockFileEx on Windows which
// serializes concurrent loads across processes). The file size is checked
// before and after the read; if it changed (a writer replaced the model
// without honoring the lock), the read is retried a few times. Writers are
// expected to either take the same lock (POSIX LOCK_EX / Windows LockFileEx)
// or replace the file atomically (write + rename).
//
// Returns false on failure; the reason is logged with MNN_ERROR.
bool ReadModelFile(const std::string& path, std::vector<uint8_t>* out);

}  // namespace mnn_engine
}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_MNN_ENGINE_MNN_MODEL_READER_H_
