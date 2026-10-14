# 7. Ten exercises, with solutions

1. **Why is only the assistant turn trained in SFT?** *Solution:* the prompt is context the model will always be given; training on it spends capacity predicting text it never has to generate, and encourages it to continue the user's turn (Volume 2, chapter 4).
2. **You have 6 training examples and perfect accuracy on them. What have you learned?** *Solution:* nothing about generalisation. Lab 50 scores 1.0 on trained prompts and 0.0 on unseen ones.
3. **Convert an Alpaca record with a non-empty `input` field into `messages`.** *Solution:* user content = `instruction + "\n\n" + input`; assistant content = `output` (`peftkit.data.format.to_messages` does exactly this).
4. **Your model works with your system prompt and fails with a shorter one. Why?** *Solution:* every training example used the same system prompt, so behaviour is conditioned on it. Vary it in training, include the empty case.
5. **Your chat model answers turn 1 well and ignores context by turn 3. Diagnose.** *Solution:* single-turn training data (lab 52: 0.33 vs 1.00 on turn 2), or truncation cutting earlier turns.
6. **You train on CoT traces and the answers get worse. Two likely causes?** *Solution:* traces were unverified (model learned to produce plausible reasoning to a wrong answer), or the traces' final-answer format differs from what your evaluator expects.
7. **Give two reasons not to use an Alpaca-derived dataset in a commercial product.** *Solution:* CC BY-NC 4.0 forbids commercial use; and the terms of the model that generated it may restrict training competing models.
8. **Your domain accuracy rose 8 points and a general benchmark fell 6. Ship?** *Solution:* usually no. Add 5-30% replay data, reduce epochs or rank, re-measure both.
9. **How do you test that your model can handle a customer it has never seen?** *Solution:* hold out entities, not rows: no example about that customer/machine/product in training at all.
10. **Training loss is 0.02 and the model is useless. What happened?** *Solution:* memorisation. Check duplicate rate, train/eval leakage, epochs, and whether the eval set shares templates with training.
