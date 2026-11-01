# Soundproof

Soundproof 音频处理算法库，提供以下算法模块：

| 模块 | 算法 |
|------|------|
| libAE_VAD | 语音活动检测（Voice Activity Detection，基于 WebRTC VAD + ISAC） |
| libAE_SRC | 采样率转换（Sample Rate Conversion，sinc 重采样器） |
| libAE_NS | 噪声抑制（Noise Suppression） |
| libAE_AGC | 自动增益控制（AGC，模拟 + 数字增益） |
| libAE_AGC2 | 自适应数字增益控制（AGC2，含 RNN 语音活动检测） |
| libAE_AECM | 声学回声消除-移动端（AEC Mobile） |
| libAE_AEC | 声学回声消除（AEC3） |
| libAE_BF | 波束成形（Beamforming，非线性波束成形器） |
| libAE_HS | 啸叫抑制（Howling Suppressor） |
| libAE_IE | 语音可懂度增强（Intelligibility Enhancer） |
| libAE_TS | 瞬态噪声抑制（Transient Suppressor） |
| libAE_DR | 去混响（Dereverberation） |
