# Train model với gradient descent

Trong hai quá trình của supervised learning (train và test) thì quá trình test đơn giản hơn vì bạn chỉ việc đưa observation vào model, nhận về label dự đoán, và tính giá trị của evaluation function trên test set. 

Quá trình train phức tạp hơn nhiều vì nó phải đảm bảo rằng model phải có khả năng tổng quát hóa. Tức là model không chỉ tốt trên train set mà còn phải tốt trên test set (chứa dữ liệu không được model nhìn thấy lúc train).

### Vì sao không dùng error rate để train model?

Để trả lời câu hỏi bạn cần phải đọc cả bài viết. Trong phần n mình chỉ nói sơ qua lý do. 

Như ta đã biết, quá trình train model về bản chất là tối ưu một hàm số. Từ kiến thức đã học từ cấp 3, ta cũng biết rằng việc tối ưu hàm số có liên quan đến đạo hàm (ví dụ như đạo hàm ở điểm cực tiểu của một hàm số bằng 0). Trong bài viết này, ta sẽ giới thiệu phương pháp tối ưu hàm số bằng gradient descent, tức là dùng gradient (đạo hàm nhiều biến) để dẫn lối cho ta đi 
đến điểm cực tiểu. Bạn có thể hình dung việc này như là đi tìm thung lũng thấp nhất trong một vùng núi non. Cách đơn giản nhất là bạn cứ thả mình lăn xuống dốc cho đến khi nào dừng lại. Gradient giống nhưng tổng lực của lực hấp dẫn và phản lực của mặt đất, sẽ kéo bạn lăn về nơi thấp hơn cho đến khi mặt đất không còn dốc nữa.

Ta cần hàm được tối thiểu hóa có gradient ở mọi nơi (hoặc chí ít là sub-gradient). Tuy nhiên khi nhìn lại một evaluation function như là error rate:
$$
e_D = \frac{1}{|D|} \sum_{(x, y) \in D} \mathbb{I}\{ f_w(x) \neq y \}
$$

Ta thấy mỗi hàm $$ \mathbb{I}\{ f_w(x) \neq y \}
$$ không có đạo hàm liên tục. Ta có thể tưởng tượng việc không có đạo hàm liên tục giống như là hàm số bị "gãy". Khi ta cho $$f_w(x)$$ đi từ $$-\infty$$ đến $$+\infty$$, hàm $$ \mathbb{I}\{ f_w(x) \neq y \}
$$ hầu hết mang giá trị 1. Chỉ đến điểm mà $$f_w(x) = y$$, hàm này độ nhiên nhảy lên giá trị 0. Điểm gãy này làm cho ta không thể áp dụng gradient descent được. Bạn thử tưởng tượng nếu đang leo núi mà rơi xuống một khe vực thì không biết đường nào mà leo lên cả. 

Vì thế người ta không tối thiểu error rate trong lúc train. Thay vào đó, người ta sẽ tối thiểu các hàm khác có tính chất:
- Có (sub)-gradient ở mọi nơi.
- Model tối thiểu hàm này cũng sẽ tối thiểu error rate.

Hàm được tối thiểu lúc train gọi là **objective function** (để phân biệt với **evaluation function** lúc test).

### Mục đích của huấn luyện

Mục đích của huấn luyện là tìm ra model . Vì model là một hàm số $f_{\theta}$ có parameter là $\theta$, theo ngôn ngữ toán học, mục đích này chính là **tìm ra tham số $\theta^*$ tối ưu sao cho trung bình loss function trên training set là nhỏ nhất**:

$$ \theta^* = \arg \min_{\theta} \mathcal{L}(\theta) = \arg \min_{\theta} \frac{1}{|D_{train}|} \sum_{(x, y) \in D_{train}} L(f_{\theta}(x), y)$$ với $D_{train}$ là training set. Kí hiệu $|D_{train}|$ nghĩa là số phần tử của training set. 

$\mathcal{L}(\theta)$ được gọi là **objective function** (hàm mục tiêu).

**Nâng cao**: để đơn giản hóa, mình đã bỏ bớt regularization trong hàm mục tiêu. Bạn có thể xem thêm về regularization [tại đây](https://ml-book-vn.khanhxnguyen.com/1_3_rlm.html).

Bài toán này là một dạng của **function optimization** (tối ưu hàm số). Ở đây vì $\theta$ không có điều kiện gì ràng buộc nên được gọi là **unconstrained optimization**. 

Nếu không có công thức trực tiếp cho $\theta^*$, ta bắt buộc phải làm nhỏ dần $\mathcal{L}(\theta)$ qua nhiều bước. Ta bắt đầu với một $\theta$ ngẫu nhiên, và tìm cách làm cho $\theta$ càng ngày càng tiến gần tới giá trị tối ưu $\theta^*$. Cách làm như vậy được gọi là một **iterative method**. Mỗi lần forward và backward chính là một bước biến đổi $\theta$ để làm $\mathcal{L}(\theta)$ nhỏ dần đi.

Nếu ai đã quen thuộc với **binary search** thì sẽ nhận ra thuật toán này cũng mang tư tưởng tương tự. Binary search thực chất là một dạng đặc biệt của function optimization với hàm được tối ưu chính là giá trị tuyệt đối giữa dự đoán hiện tại và giá trị cần tìm. 