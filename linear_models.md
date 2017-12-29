# Model tuyến tính (linear models)

Để hiểu rõ về "deep learning", đầu tiên ta cần phải biết thế nào là "không deep learning"? Những model như thế nào gọi là "không deep" và chúng có những tính chất gì, bài viết này sẽ trả lời cho câu hỏi đó. 

Deep learning chỉ những model có nhiều lớp và phát triển theo chiều sâu. Trái ngược với chúng là những model chỉ có một lớp và phát triển theo chiều dọc. Đó chính là các **linear models** (model tuyến tính). Linear learning là nền tảng cho deep learning bởi vì các model deep learning chỉ đơn giản là một khối gồm nhiều model tuyến tính được chồng lên với nhau và được kết nối bằng các phép tính không tuyến tính (non-linear). 

Như thường lệ, ta gọi $$x$$ là observation, $$y$$ là label đúng và $$\hat{y}$$ là label được dự đoán bởi model. Các model tuyến tính $$f_{(W, b)}^{lin}$$ với parameter là $$W$$ và $$b$$ đều có dạng như sau:
$$
    f_{(W, b)}^{lin}(x) = \hat{y} = W \cdot x + b
$$ với dấu $$\cdot$$ thể hiện phép nhân ma trận. 

Ở đây, $$W$$ được gọi là **weight** và $$b$$ được gọi là **bias**. Cả hai đều là parameter của model và đều được tính toán ra (hay còn gọi là "học") trong quá trình train model. 

Gỉa sử $$x$$ là một vector có $$m$$ chiều và $$\hat{y}$$ là một vector có $$n$$ chiều. Khi đó, $$W$$ bắt buộc phải là một ma trận có $$n$$ dòng $$m$$ cột và $$b$$ bắt buộc phải là một vector $$n$$ chiều. Trong ngôn ngữ toán học, ràng buộc này được viết là "$$W \in \mathbb{R}^{n \times m}, b \in \mathbb{R}^n$$". 

Linear models **chỉ đơn giản thực hiện một phép nhân ma trận sau đó là một phép cộng vector đối với input và trả về vector kết quả**. Tuy đơn giản như vậy nhưng trong một thời gian dài, linear models đã đóng góp rất lớn vào thành công của các ứng dụng machine learning. Một ví dụ điển hình là Google Ads. Google có một model để dự đoán số lượng click vào mỗi quảng cáo, và dựa vào đó để đưa ra quảng cáo phù hợp với bạn. Model được Google sử dụng là một model logistic regression model. Thật bất ngờ khi đằng sau công nghệ hàng đầu thế giới như vậy chỉ là hai phép tính đại số cơ bản đúng không? 

### Một ví dụ đơn giản

Hãy tưởng tượng rằng ta đang muốn đoán xem hôm nay trời có mưa hay không dựa vào hai yếu tố: nhiệt độ và độ ẩm. Đây là một bài toán *binary classification* bởi vì ta cần phân loại các  observation vào một trong *hai* nhãn (trường hợp này là "mưa" hay "không mưa"). Ta thiết kế một linear model để giải bài toán này như sau: 
- *Observation* $$x$$: vector hai chiều, là các ghi nhận về nhiệt độ và độ ẩm (một chiều thể hiện nhiệt độ, chiều còn lại thể hiện độ ẩm)
- *Label đúng* $$y$$ và *label dự đoán* $$\hat{y}$$: vector một chiều thể hiện dự đoán mưa hay không mưa. Vì chỉ có một chiều nên các vector này thực ra chỉ là số thực. Label chỉ mang một trong hai giá trị 0 (không mưa) và 1 (mưa). Đối với label dự đoán, ta quy ước là nếu $$\hat{y} \geq 0$$ model dự đoán là trời mưa, và ngược lại nếu $$\hat{y} < 0$$ model dự đoán là không mưa. 

Giả sử model có $$W = \begin{pmatrix}
    -0.2\\\ 
    0.5
\end{pmatrix}
$$ và $$b = \begin{pmatrix}
    3\\\ 
    -40
\end{pmatrix}
$$. Với một ngày nóng và khô, nhiệt độ 35 độ C và độ ẩm 50%, dự đoán của model sẽ là
$$
\begin{pmatrix}
    -0.2\\\ 
    0.5
\end{pmatrix}
\begin{pmatrix}
    35\\\ 
    50
\end{pmatrix}
 + \begin{pmatrix}
    3\\\ 
    -40
\end{pmatrix}
 = -6 + (-15) = -21 < 0
$$
, tức là dự đoán trời không mưa. Ngược lại, với một ngày nhiệt độ trung bình nhưng rất ẩm, nhiệt độ 28 độ C và độ ẩm 90%, dự đoán của model sẽ là
$$
(-0.2, -0.5)^{\top}(28, 90) + (3, -40) = -2.6 + 5 = 2.4 \geq 0
$$, tức là dự đoán trời không mưa. 

TODO: hình minh họa

### Linear transformation 

Sở dĩ linear models có tên như vậy bởi vì về bản chất chúng là những linear transformation (biến đổi tuyến tính). Linear transformation $$f$$ thỏa mãn tính chất sau:
$$
    f(a_1 x_1 + a_2 x_2) = a_1 f(x_1) + a_2 f(x_2)
$$, tức là áp dụng hàm lên một linear combination (tổ hợp tuyến tính) của các biến cũng giống như áp dụng cùng linear combination đó lên kết quả của hàm này trên các biến.

**Câu hỏi**: giải thích vì sao gradient (đạo hàm) là một linear transformation?

Trong linear algebra (đại số tuyến tính), mỗi linear transformation được đặc trưng bởi một ma trận $$W$$ và đều có dạng $$f(x) = W\cdot x$$. Tùy vào ma trận này, việc áp dụng một linear transformation lên một vector sẽ biến đổi vector theo các cách khác nhau. Sau đây là một số dạng ma trận và linear transformation đặc biệt:

Permutation matrix: tráo đổi của các chiều của một vector 
Ví dụ: 
$$ W \cdot x = 
\begin{pmatrix}
    0 & 1 & 0\\\ 
    1 & 0 & 0\\\
    0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
    x_1\\\ 
    x_2\\\
    x_3
\end{pmatrix}
= 
\begin{pmatrix}
    x_2\\\ 
    x_1\\\
    x_3
\end{pmatrix}
$$ 

Rotation matrix: xoay vector ngược chiều kim đồng hồ một góc $$\theta$$ 
$$ W \cdot x = 
\begin{pmatrix}
    \cos \theta & -\sin \theta\\\ 
    \sin \theta & \cos \theta 
\end{pmatrix}
\begin{pmatrix}
    x_1\\\ 
    x_2
\end{pmatrix}
$$ 

Scaling matrix: thay đổi tỉ lệ giữa các chiều 
$$ W \cdot x = 
\begin{pmatrix}
    d_1 & 0 & 0 \\\ 
    0 & d_2 & 0 \\\
    0 & 0 & d_3
\end{pmatrix}
\begin{pmatrix}
    x_1\\\ 
    x_2\\\
    x_3
\end{pmatrix}
= 
\begin{pmatrix}
    d_1 x_1\\\ 
    d_2 x_2\\\
    d_3 x_3
\end{pmatrix}
$$ 

**Nâng cao**: ta thấy là so với công thức của linear function, trong các linear models còn có thêm bias $$b$$. Sự xuất hiện của $$b$$ có phải làm cho linear trái với định nghĩa của linear transformation hay không? Câu trả lời là không. Một cách chính xác nhất, các hàm có dạng $$f(x) = W\cdot x + b$$ như linear models không phải là linear function mà là **affine function**. Điểm khác giữa affine function và linear function chỉ là ở chỗ affine có thêm bias. Tuy nhiên, các affine function vẫn là một dạng của linear transformation. Ta có thể biến đổi một affine function thành linear function bằng một mẹo nhỏ như sau: định nghĩa $$W'$$ có $$n$$ dòng và $$m + 1$$ cột với $$m$$ cột đầu là của $$W$$ và cột cuối cùng là vector $$b$$; với mỗi input $$x$$, định nghĩa $$x'$$ là vector $$m + 1$$ chiều với $$m$$ chiều đầu là của $$x$$ và chiều cuối cùng bằng 1. Khi đó, linear function $$W' \cdot x'$$ sẽ tương đường với affine function $$W \cdot x + b$$. 

### Huấn luyện

### Kết hợp với các loss function 






