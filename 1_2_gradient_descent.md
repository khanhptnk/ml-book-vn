# Gradient descent

Trong hai quá trình của supervised learning, train và test, đã được giới thiệu ở bài trước thì quá trình test đơn giản hơn vì bạn chỉ việc đưa observation vào model, nhận về label dự đoán, và tính giá trị của evaluation function trên test set. Quá trình train phức tạp hơn vì nó phải đảm bảo rằng model phải có khả năng tổng quát hóa trên dữ liệu mới. Trong một bài toán machine learning, độ tốt của model trên test set là thước đo chính xác nhất và quan trọng nhất về khả năng học của model. Vì thế quá trình train phải đảm bảo model dự đoán tốt trên test set.

### Objective function

Để dự đoán tốt nhất trên test set, cách đơn giản nhất là tìm model dự đoán tốt nhất trên train set, và *hy vọng* rằng nó cũng sẽ tốt trên test set.
Vì thế, ở bài trước ta phát biểu rằng quá trình train là tìm ra model tối thiểu hóa evaluation function trên train set. Tuy nhiên, cách làm thực sự đang đơn giản hóa vấn đề và thực tế không hiệu quả. Ta sẽ có hai thay đổi để làm nó tốt hơn.

Thứ nhất, quá trình train là việc tìm ra model dự đoán "khá" chính xác trên train set. Vì sao là "khá" chính xác mà không phải là chính xác hoàn toàn? Model dự đoán hoàn chính xác trên train set có thể dự đoán rất tệ trên test set, nếu hai set này rất khác nhau. Điều giống như việc bạn bị "trật tủ" khi đi vậy (ôn một đằng đề ra một kiểu). Thế nên ta cần chừa ra một khoảng "không chính xác" để bù lại model dự đoán trên dữ liệu chưa nhìn thấy tốt hơn (gọi là khả năng tổng quát hóa). 

Tuy nhiên, để đơn giản, bây giờ ta cứ hiểu là cần tìm một model dự đoán tốt nhất trên train set. Ta sẽ có nhiều điều chỉnh để model tổng quát hóa tốt hơn sau.

Với cách hiểu này, việc train chỉ đơn là tìm ra model tối thiểu hóa evaluation function được tính trên train set. 

$$
\min_w \mathcal{L}_{D_{train}}(f_w)
$$

Ví dụ nếu loss function là error rate, thì mục tiêu của ta là tìm ra model đoán đúng nhiều label nhất trên train set. Nghe rất hợp lý phải không nào! Nhưng làm sao để tìm ra? 

Tối ưu những loss function như error rate rất khó. Trên thực tế người ta sẽ tối ưu những dạng loss function khác? Những hàm đó là gì? Chúng có tính chất gì giúp tìm ra model tối ưu dễ dàng hơn? Phương nào dùng để tìm ra model tối ưu cho những hàm đó? Chúng ta sẽ cùng tìm hiểu trong bài này :)

### Objective function



### Mục đích của huấn luyện

Mục đích của huấn luyện là tìm ra model . Vì model là một hàm số $f_{\theta}$ có parameter là $\theta$, theo ngôn ngữ toán học, mục đích này chính là **tìm ra tham số $\theta^*$ tối ưu sao cho trung bình loss function trên training set là nhỏ nhất**:

$$ \theta^* = \arg \min_{\theta} \mathcal{L}(\theta) = \arg \min_{\theta} \frac{1}{|D_{train}|} \sum_{(x, y) \in D_{train}} L(f_{\theta}(x), y)$$ với $D_{train}$ là training set. Kí hiệu $|D_{train}|$ nghĩa là số phần tử của training set. 

$\mathcal{L}(\theta)$ được gọi là **objective function** (hàm mục tiêu).

**Nâng cao**: để đơn giản hóa, mình đã bỏ bớt regularization trong hàm mục tiêu. Bạn có thể xem thêm về regularization [tại đây](https://ml-book-vn.khanhxnguyen.com/1_3_rlm.html).

Bài toán này là một dạng của **function optimization** (tối ưu hàm số). Ở đây vì $\theta$ không có điều kiện gì ràng buộc nên được gọi là **unconstrained optimization**. 

Nếu không có công thức trực tiếp cho $\theta^*$, ta bắt buộc phải làm nhỏ dần $\mathcal{L}(\theta)$ qua nhiều bước. Ta bắt đầu với một $\theta$ ngẫu nhiên, và tìm cách làm cho $\theta$ càng ngày càng tiến gần tới giá trị tối ưu $\theta^*$. Cách làm như vậy được gọi là một **iterative method**. Mỗi lần forward và backward chính là một bước biến đổi $\theta$ để làm $\mathcal{L}(\theta)$ nhỏ dần đi.

Nếu ai đã quen thuộc với **binary search** thì sẽ nhận ra thuật toán này cũng mang tư tưởng tương tự. Binary search thực chất là một dạng đặc biệt của function optimization với hàm được tối ưu chính là giá trị tuyệt đối giữa dự đoán hiện tại và giá trị cần tìm. 



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

