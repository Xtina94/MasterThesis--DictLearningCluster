In this folder I modified the initial algorithm in two main ways:
1. I added to the graph learning part also the dictionary learning part, optimizing alternatevly between the two of them;
2. In the dictionary learning part I used the constraints over the kernels in the optimization function, to be precise in the constrain definition part;
3. I started the optimization supposing a random heat kernels and and optimizing over the graph learning first;

Let's see if this gives better results than starting the optimization cycle from the dictionary learning part;