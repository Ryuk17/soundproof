# SoundProof 项目约定

## 1. 项目简介

SoundProof 是跨平台音频处理算法库（Linux / Windows / 嵌入式），以 C ABI 对外提供 `libAE_xxx` 静态库。算法分为两类：

- **传统 DSP（WebRTC 系）**：VAD、SRC、NS、AGC、AGC2、AECM、AEC3、BF、HS、IE、TS、DR
- **MNN 深度学习算法**：KWS（zipformer2 唤醒词）、ASR（transducer / paraformer）、SR（说话人识别）

MNN 相关算法只在 `WITH_MNN=ON` 时构建，其余算法默认不依赖 MNN。C++ 内部实现使用 C++20；对外头文件保持纯 C（仅 `stdint.h`），可被 C/C++ 调用。

## 2. 目录结构（仅 git 跟踪内容）

```
CMakeLists.txt          单文件构建入口（库、测试、依赖下载）
cmake/                  mnn.cmake（MNN 3.6.1）、knf.cmake（kaldi-native-fbank 1.21.1）
toolchains/             MinGW / aarch64 / armhf 交叉编译工具链文件
README.md / LICENSE

soundproof/
├── interface/          对外 C 接口
│   ├── audio_engine_def.h     统一错误码、AUDIO_ENGINE_API、路径长度
│   ├── audio_engine_log.h     接口层日志宏（LOG_DEBUG/INFO/WARN/ERROR）
│   └── audio_engine_<algo>.h/.cpp   每个算法的 C 接口与实现
├── feature/            共享 fbank 特征提取（参数化，KWS/ASR/SR 共用）
├── utils/              共享工具（symbol_table、WAV、logging、aligned_malloc 等）
├── model/              transducer 模型封装（encoder/decoder/joiner）
├── mnn_engine/         通用 MNN 推理引擎（内部源码，不产出独立库）
├── dsp/                WebRTC 风格 DSP 基础实现
└── core/               算法实现
    ├── acoustic_echo_cancellation3/       AEC3
    ├── acoustic_echo_cancellation_mobile/ AECM
    ├── automatic_gain_control/            AGC
    ├── automatic_gain_control2/           AGC2（含 RNN VAD）
    ├── beamforming/                       BF
    ├── dereverberation/                   DR
    ├── howling_suppressor/                HS
    ├── intelligibility_enhancer/          IE
    ├── keywords_spotting/                 KWS
    ├── noise_suppressor/                  NS
    ├── sample_rate_conversion/            SRC
    ├── speech_recognition/                ASR（后端抽象、端点、识别前端）
    ├── speaker_recognition/               SR（后端抽象、注册库、识别前端）
    ├── transient_suppressor/              TS
    ├── voice_activity_detection/          VAD（WebRTC + Silero）
    └── tests/                             算法内部参考测试（未纳入当前 CMake）

tests/                   C 接口测试：test_ae_<algo>.cc、test_mnn_engine.cc
assets/                  测试音频与算法输出基线
models/                  模型获取脚本 download.sh 与 rnnoise 源码；
                         其余模型资产不入库，由 download.sh 获取
```

## 3. 接口设计要求（强制）

### 3.1 文件与符号

- 公共接口只放在 `soundproof/interface/audio_engine_<algo>.h/.cpp`。
- 头文件使用 `extern "C"`，所有函数加 `AUDIO_ENGINE_API`，只包含 `<stdint.h>` 和 `audio_engine_def.h`，禁止暴露 C++ 类型。
- 实现文件包含 `audio_engine_log.h`，禁止在 C ABI 中抛出异常、`exit()` 或 `abort()`。

### 3.2 生命周期

每个算法提供统一函数集：

```c
XxxHandle AudioEngine_Xxx_Create(void);
int AudioEngine_Xxx_Destroy(XxxHandle handle);
int AudioEngine_Xxx_Init(XxxHandle handle, const XxxInitConfig* init_config);
int AudioEngine_Xxx_SetParam(XxxHandle handle, const XxxRtConfig* rt_config);
int AudioEngine_Xxx_ResetParam(XxxHandle handle, const XxxRtConfig* rt_config);
int AudioEngine_Xxx_Process(XxxHandle handle, const XxxInput_t* input, XxxResult_t* result);
int AudioEngine_Xxx_Deinit(XxxHandle handle);
int AudioEngine_Xxx_Reset(XxxHandle handle);
```

- `Create/Destroy`：句柄内存生命周期；`Destroy` 内部先释放实例。
- `Init/Deinit`：算法实例生命周期；`Deinit` 后句柄仍可用，可再次 `Init`。
- `SetParam`：增量更新；`ResetParam`：先恢复默认值再应用。
- `Reset`：重建算法实例/清空流状态，保留配置。
- `Init` 必须为运行参数写入默认值，防止调用方跳过 `SetParam`。

### 3.3 Process 统一签名

- **除 handle 外只允许两个参数**：输入结构体 `const XxxInput_t*`、输出结构体 `XxxResult_t*`。禁止继续使用散参数。
- 输入结构体字段：输入缓冲指针、样本数/帧长及算法特有输入；输出结构体字段：输出缓冲指针、容量、实际输出长度及结果标志。
- 输入结构体字段名统一：`audio_in`、`in_samples`、`audio_out`、`max_out_samples`、`out_samples`；算法特有字段沿用现有命名（如 `nearend_in`、`trigger`）。
- 现有算法：VAD、SRC（Resample）、NS、AGC、AGC2、AECM、AEC、BF、HS、IE、TS、DR、KWS、ASR、SR 均已按此约定。

### 3.4 命名约定

| 类型 | 命名 |
|---|---|
| 句柄 | `XxxHandle` |
| 初始化配置 | `XxxInitConfig` |
| 运行参数 | `XxxRtConfig` |
| Process 输入 | `XxxInput_t` |
| Process 输出 | `XxxResult_t` |
| 算法库 | `libAE_XXX`（CMake target `AE_XXX`） |

### 3.5 校验与日志

`Process` 校验顺序固定：

1. `handle` 为空 → `INVALID_HANDLE`
2. `input` / `result` 为空 → `NULL_POINTER`
3. 输入/输出缓冲指针为空 → `NULL_POINTER`
4. 未初始化 → `NOT_INITIALIZED`
5. 参数范围（采样率、帧长、容量、阈值等）→ `INVALID_PARAM`
6. 算法执行失败 → 对应具体错误码

- 每个公共 API 入口 `LOG_DEBUG`；所有失败路径 `LOG_WARN`/`LOG_ERROR`，日志必须包含 API 名与具体原因（例如 `audio too short (10 frames < 50)`、`speaker 'alice' already exists`）。
- 返回码必须使用最具体的一项，不允许用 `PROCESS_FAILED` 掩盖已知原因。

### 3.6 线程与平台

- 句柄默认**非线程安全**；同一句柄不可跨线程并发调用。MNN 引擎内部对 `Forward` 做了串行化，但算法实例状态仍需调用方保证单线程使用。
- 固定帧长/采样率约束写在头文件注释中，`Init` 与 `Process` 均需校验。
- 新增算法必须同时提供 C 接口、CMake `option(BUILD_AE_XXX)`、`test_ae_xxx.cc`，并更新 README 模块表。

### 3.7 MNN 算法额外约定

- `mnn_engine`、`feature`、`model`、`utils` 为共享源码，直接编译进消费方算法库，不产出独立库。
- MNN 算法只在 `WITH_MNN=ON` 时构建；`WITH_MNN=OFF` 时 CMake 打印跳过，不影响其它算法。
- 模型类型通过 metadata 自动识别；`model_type` 字段可显式覆盖。
- 测试在模型/音频缺失时必须打印 `SKIP` 并返回成功，不能依赖未入库资产。

## 4. 错误码要求

错误码唯一来源：`soundproof/interface/audio_engine_def.h` 的 `AudioEngineStatus`。

### 4.1 区间分配

| 区间 | 用途 |
|---|---|
| `0x00 – 0xFF` | 公用错误码（所有算法共享） |
| `0x100 – 0x1FF` | SR（Speaker Recognition）专用 |
| `0x200 – 0x2FF` | 预留给下一个需要专用错误码的算法 |
| 后续算法 | 依次分配 `0x300`、`0x400` … |

规则：

- 已发布的错误码**禁止重新编号或复用**；新增公用码追加到 `0x00–0xFF` 末尾。
- 新算法专用码必须在自己的区间内追加，并显式使用十六进制字面量。
- 不允许使用区间外的值；不允许把算法专用码放入公用区间。

### 4.2 当前错误码

公用：

```c
AUDIO_ENGINE_SUCCESS                    = 0x00
AUDIO_ENGINE_ERR_INIT_FAILED            = 0x01
AUDIO_ENGINE_ERR_SET_PARAM_FAILED       = 0x02
AUDIO_ENGINE_ERR_INVALID_HANDLE         = 0x03
AUDIO_ENGINE_ERR_NULL_POINTER           = 0x04
AUDIO_ENGINE_ERR_NOT_INITIALIZED        = 0x05
AUDIO_ENGINE_ERR_PROCESS_FAILED         = 0x06
AUDIO_ENGINE_ERR_INVALID_PARAM          = 0x07
AUDIO_ENGINE_ERR_MODEL_LOAD_FAILED      = 0x08
AUDIO_ENGINE_ERR_UNSUPPORTED_MODEL      = 0x09
AUDIO_ENGINE_ERR_AUDIO_TOO_SHORT        = 0x0A
AUDIO_ENGINE_ERR_FEATURE_FAILED         = 0x0B
AUDIO_ENGINE_ERR_INFERENCE_FAILED       = 0x0C
AUDIO_ENGINE_ERR_EMBEDDING_DIM_MISMATCH = 0x0D
```

SR 专用：

```c
AUDIO_ENGINE_ERR_SR_SPEAKER_EXISTS      = 0x100
AUDIO_ENGINE_ERR_SR_SPEAKER_NOT_FOUND   = 0x101
AUDIO_ENGINE_ERR_SR_BUFFER_TOO_SMALL    = 0x102
```

### 4.3 使用要求

- 每个失败路径必须同时**记录原因**并返回对应错误码。
- MNN 后端错误映射：模型加载 → `MODEL_LOAD_FAILED`；模型类型/结构不支持 → `UNSUPPORTED_MODEL`；特征提取 → `FEATURE_FAILED`；前向失败 → `INFERENCE_FAILED`。
- 输入音频过短 → `AUDIO_TOO_SHORT`；输出缓冲不足 → 通用或算法专用 `BUFFER_TOO_SMALL`。
- 新增错误码时同步更新本文件第 4.2 节。

## 5. 构建与测试

```bash
# 默认（无 MNN）
cmake -S . -B build -G "MinGW Makefiles" \
      -DCMAKE_TOOLCHAIN_FILE=toolchains/x86_64-windows.cmake
cmake --build build -j

# 含 KWS / ASR / SR（MNN）
cmake -S . -B build -G "MinGW Makefiles" \
      -DCMAKE_TOOLCHAIN_FILE=toolchains/x86_64-windows.cmake -DWITH_MNN=ON
cmake --build build -j 1        # MinGW 内存有限时建议 -j1
ctest --test-dir build --output-on-failure
```

构建选项：`BUILD_AE_<ALGO>` 控制各算法库与测试；`WITH_MNN` 控制 MNN 运行时；`WITH_MNN_LLM` 启用 MNN LLM 支持（隐含 `WITH_MNN`）。模型资产通过 `models/download.sh` 获取，不入库。
