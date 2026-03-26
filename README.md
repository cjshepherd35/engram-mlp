Engram-mlp is a modification of the work made in the paper by Deepseek called Conditional memory via scalable lookup table. In my version instead of finding the embedding vector in a 
lookup table it sends the context into an mlp, since an mlp is a way to store memories in a compressed format. I applied it to Karpathy's nanogpt with modifications to use byte pair encoding 
also learned from Karpathy's youtube channel. 
  The first file is a recreation of Deepseeks Engram applied to Nanogpt built using Antigravity, then the second file is my idea of the Engram-mlp also built using Gemini. The inputs.txt file
is a download of the tiny shakespeare dataset. 
