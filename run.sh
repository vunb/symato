[ -f 20B_tokenizer.json ] || wget https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4neo/20B_tokenizer.json
[ -f RWKV-4-Pile-169M-20220807-8023.pth ] || wget https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth
python3 model_run_f32.py
