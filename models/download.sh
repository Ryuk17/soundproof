apt update && apt install aria2 -y

wget https://hf-mirror.com/hfd/hfd.sh 

chmod +x hfd.sh

export HF_ENDPOINT=https://hf-mirror.com

./hfd.sh taobao-mnn/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20 --local-dir sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20

./hfd.sh taobao-mnn/Qwen3.5-0.8B-MNN --local-dir llm/Qwen3.5-0.8B-MNN

./hfd.sh taobao-mnn/Qwen3-Embedding-0.6B-MNN --local-dir embedding/Qwen3-Embedding-0.6B-MNN

./hfd.sh Jeryuk/kokoro-v1_0-MNN --local-dir tts/kokoro-v1_0-MNN

./hfd.sh Jeryuk/smart-turn-v3.2-gpu-MNN --local-dir turn/smart-turn-v3.2-gpu.mnn

./hfd.sh Jeryuk/silero_vad-MNN --local-dir vad/Jeryuk/silero_vad-MNN

./hfd.sh Jeryuk/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common-MNN --local-dir speaker_recognition/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common-MNN

./hfd.sh Jeryuk/kws-zipformer-wenetspeech-3.3M-MNN --local-dir speaker_recognition/kws-zipformer-wenetspeech-3.3M-MNN
