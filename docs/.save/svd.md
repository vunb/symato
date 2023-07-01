Tìm hiểu:
- ma trận chéo hóa được
- continuous ODE (a bit similar to State Space Models)

https://machinelearningcoban.com/2017/06/07/svd
Một ma trận vuông 
A ∈ R^{n×n} được gọi là chéo hoá được (diagonalizable) nếu tồn tại ma trận đường chéo D và ma trận khả nghịch P sao cho: `A = P D P^-1` (1). Nhân cả 2 vế của (1) với P ta có: `A P = P D` (2).

Cách phân tích một ma trận vuông thành nhân tử như (1) còn được gọi là Eigen Decomposition.

Một điểm quan trọng là cách phân tích như (1) chỉ được áp dụng với ma trận vuông và không phải lúc nào cũng tồn tại. __Nó chỉ tồn tại nếu ma trận A có n vector riêng độc lập tuyến tính__, vì nếu không thì không tồn tại ma trận P khả nghịch.

Việc phân tích một ma trận ra thành tích của nhiều ma trận đặc biệt khác (Matrix Factorization hoặc Matrix Decomposition) mang lại nhiều ích lợi quan trọng mà các bạn sẽ thấy:
- giảm số chiều dữ liệu, 
- nén dữ liệu, tìm hiểu các đặc tính của dữ liệu, 
- giải các hệ phương trình tuyến tính, clustering, 
- và nhiều ứng dụng khác.
Recommendation System cũng là một trong rất nhiều ứng dụng của [Matrix Factorization](https://machinelearningcoban.com/2017/05/31/matrixfactorization).

## Hungry Hungry Hippos: Towards Language Modeling with State Space Models
https://arxiv.org/abs/2212.14052
