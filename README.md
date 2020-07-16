# NLP_coding


### 1. [Bert](https://github.com/google-research/bert)
#### 1.1 OrderedDict
https://docs.python.org/zh-cn/3/library/collections.html
OrderedDict和namedtuple学习一下使用。
OrderedDict比Dict的映射性能低一些，它适合在需要记住插入顺序的场景中使用。
这份代码里OrderedDict是在讲数据变成tf.train.Example实例的使用被用到。就当dict使用了，往key里面传value。原来tf.train.Features()构造函数里面可以传OrderedDict。
