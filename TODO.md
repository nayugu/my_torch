Big problems and reasons to create v2

- Current traversal of computation graph is extremely inefficient. It transverses every single path from end-to-end.
- Current code uses only multiplication to send gradients backwards, but different operations need different rules (e.g. matmul, softmax). Current implementation has bugs.