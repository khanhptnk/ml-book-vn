# Thuật toán supervised learning tổng quát

Sau khi đi qua RLM, ta đi được phần lớn quãng đường để xây dựng một thuật toán supervised learning hoàn chỉnh. Bài viết này sẽ đưa các bạn đi nốt phần đường còn lại. 

### Tập phát triển

Khi bình thường hóa tham số xuất hiện trong hàm mục tiêu, vấn đề đặt ra là ta không nhất thiết phải đặt giá trị của hằng số bình thường hóa $$\lambda$$ giống nhau cho mọi bài toán. Hơn nữa, ngoài $$\lambda$$, còn có nhiều hyperparameter khác ta cần phải lựa chọn (như bậc của đa thức). Làm sao để chọn được tập giá trị tối ưu cho các hyperparameter với từng bài toán?

Ta có thể làm như sau: ta chọn một tập giá trị của các hyperparameter, huấn luyện để tìm ra một model rồi đo độ tốt của nó trên tập kiểm tra. Ta tiếp tục lặp lại quá trình này với nhiều tập giá trị hyperparameter khác nhau. Sau nhiều lần thử chọn như vậy, ta chọn tập giá trị nào cho độ sai sót thấp nhất trên tập kiểm tra.

Cẩn thận! Khi dùng tập kiểm tra để xác định hyperparameter, ta đã vi phạm **nguyên tắc train-test độc lập** đã nêu ra ở đầu chương. Nói một cách đơn giản là ta đã sử dụng tập kiểm tra để chọn model. Để khắc phục điều này, ta cần đến một "tập kiểm tra thứ hai", chỉ chuyên dùng để tinh chỉnh các hyperparameter và không dùng để đưa thông báo cuối cùng về độ tốt của model. Ta gọi đấy là **tập phát triển** (development set).

Trong bài viết trước, vì chưa nhắc giới thiệu khái niệm tập phát triển nên định nghĩa early stopping của mình cũng đã vi phạm quy tắc train-test độc lập. Cụ thể, vì thời điểm dừng huấn luyện phụ thuộc vào độ sai sót trên tập kiểm tra, mà model cuối cùng nhận được lại phụ thuộc vào thời điểm dừng huấn luyện, suy ra tập kiểm tra đã gián tiếp chỉ định model cuối cùng. Sau khi biết đến tập phát triển, để áp dụng early stopping một cách đúng đắn, thì ta chỉ việc thay learning curve trên tập kiểm tra bằng learning curve trên tập phát triển.

![](http://khanhxnguyen.com/wp-content/uploads/2016/06/early-stopping-2.png)

Trong nghiên cứu, tỉ lệ train:dev:test thường được dùng đó là 7:1:2.

Và phút giây được mong chờ đã đến, ta đã tìm được một thuật toán hiệu quả cho supervised learning:

Sử dụng tập phát triển để tinh chỉnh hyperameter của model: với mỗi tập giá trị của các hyperparameter (bao gồm cả $$\lambda$$):
Huấn luyện: tìm $$w$$ để tối thiểu hóa $$\mathcal{L}_{D_{train}}(w) + \lambda R(w)$$. Trong quá trình huấn luyện, theo dõi learning curve để áp dụng early stopping. 
Đánh giá trên tập phát triển: thông báo độ tốt trên tập phát triển là $$\mathcal{L}_{D_{dev}}(w)$$. 
Đánh giá trên tập kiểm tra: với model $$w^*$$ cho kết quả tốt nhất ở bước 1, thông báo độ tốt cuối cùng trên tập kiểm tra là $$\mathcal{L}_{D_{test}}(w^*)$$.
