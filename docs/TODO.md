1.  MDLSTMs à la gpu
2.  Padding does/should not alter image size
3.  Scaling by line heihgt and (somewhat) flexible width, which is filled with 0s

---

1.  Attention
    1.1 Idea: Stack Row-Wise and run through BLSTM & then mult with encoder output
2.  Increase parameters
3.  Resnet instead of CNN
4.  Different datasets
5.  mdlstm email aachen

--- IDEAS FOR line htr

1.  Remove bias from cnn
2.  Batch Norm w/wo -> maybe now is_train works
3.  switch row/col back
