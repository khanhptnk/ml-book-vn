### Thuật toán supervised learning tổng quát

1. **Sử dụng tập phát triển để tinh chỉnh hyperameter của model**: với mỗi tập giá trị của các hyperparameter (bao gồm cả $$\lambda$$):

    a.** Huấn luyện**: tìm $$w$$ để tối thiểu hóa $$\mathcal{L}_{D_{train}}(w) + \lambda R(w)$$ trên tập huấn luyện. Trong quá trình huấn luyện, theo dõi learning curve để áp dụng early stopping. 
    
    b. **Đánh giá trên tập phát triển**: thông báo độ tốt trên tập phát triển là $$\mathcal{L}_{D_{dev}}(w)$$. 

2. **Đánh giá trên tập kiểm tra**: với model $$w^*$$ cho kết quả tốt nhất ở bước 1, thông báo độ tốt cuối cùng trên tập kiểm tra là $$\mathcal{L}_{D_{test}}(w^*)$$.
