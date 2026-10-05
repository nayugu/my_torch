### Big problems and reasons to create v2

1. Current traversal of computation graph is extremely inefficient. It transverses every single path from end-to-end. Suggest looking into topological sort.

2.  Current code uses only multiplication to send gradients backwards, but different operations need different rules (e.g. matmul, softmax). Current implementation has bugs. By analogy, the current design is like a post office with one sorting rule applied to every package. It works until odd-shaped packages show up, and then you keep adding exceptions. The fix is to have each package come with its own delivery instructions.