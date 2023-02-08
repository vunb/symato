# Illustrating Reinforcement Learning from Human Feedback
https://huggingface.co/blog/rlhf

- Đào tạo trước một mô hình ngôn ngữ (LM),
- thu thập dữ liệu và đào tạo một mô hình khen thưởng, và
- tinh chỉnh LM với học tăng cường.

## Reward model training
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png)

## Fine-tuning with RL
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png)

 Tinh chỉnh một số hoặc tất cả các tham số của bản sao của LM ban đầu bằng policy-gradient RL algorithm, Tối ưu hóa Chính sách Gần nhất (PPO: Proximal Policy Optimization). Các tham số của LM bị đóng băng vì việc tinh chỉnh toàn bộ mô hình tham số 10B hoặc 100B+ rất tốn kém (để biết thêm thông tin, hãy xem Thích ứng với thứ hạng thấp (LoRA) cho LM hoặc Sparrow LM từ DeepMind).

Đầu tiên chúng ta hãy xây dựng nhiệm vụ tinh chỉnh này như một bài toán RL. Đầu tiên, chính sách là một mô hình ngôn ngữ nhận lời nhắc và trả về một chuỗi văn bản (hoặc chỉ là phân phối xác suất trên văn bản). __Không gian hành động__ của chính sách này là tất cả các tokens trong bộ từ vựng của mô hình ngôn ngữ (khoảng 50k) và __không gian quan sát__ là các chuỗi tokens đầu vào có thể, cũng khá lớn (vocab_size ^ tokens_len). __Hàm phần thưởng__ là sự kết hợp của preference model và một ràng buộc đối với sự thay đổi chính sách.

# Fine-Tuning Language Models from Human Preferences
https://arxiv.org/pdf/1909.08593.pdf

# LoRA: Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685

# TRL - Transformer Reinforcement Learning
https://github.com/lvwerra/trl

# Offline RL for Natural Language Generation with Implicit Language Q Learning
https://sea-snell.github.io/ILQL_site
