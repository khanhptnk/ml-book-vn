# Gradient descent (in progress)

Trong hai quá trình của supervised learning, train và test, thì quá trình test đơn giản hơn vì bạn chỉ việc đưa observation vào model, nhận về label dự đoán, và tính giá trị của evaluation function trên test set. Quá trình train phức tạp hơn vì nó phải đảm bảo rằng model phải có khả năng dự đoán tốt trên test set (là dữ liệu không được model nhìn thấy lúc train). 

### Objective function

Để dự đoán tốt nhất trên test set, cách đơn giản nhất là tìm model dự đoán tốt nhất trên train set, và *hy vọng* rằng nó cũng sẽ tốt trên test set.
Vì thế, ở bài trước ta phát biểu rằng quá trình train là tìm ra model tối thiểu hóa evaluation function trên train set. Tuy nhiên, cách làm thực sự đang đơn giản hóa vấn đề và thực tế không hiệu quả. Ta sẽ có hai thay đổi để làm nó tốt hơn.

Thứ nhất, quá trình train là việc tìm ra model dự đoán "khá" chính xác trên train set. Vì sao là "khá" chính xác mà không phải là chính xác hoàn toàn? Các bạn thấy là không có điều gì đảm bảo model dự đoán hoàn chính xác trên train set cũng dự đoán tốt trên test set cả. Thậm chí nó có thể dự đoán rất tệ nếu test set rất khác với train set. Điều giống như việc bạn bị "trật tủ" khi đi vậy (ôn một đằng đề ra một kiểu). Ví dụ bạn train một model để dịch từ Anh-Việt, thì không ai đem model đó để test xem nó dịch Anh-Pháp tốt đến đâu. Thậm chí nếu model dịch Anh-Việt chỉ được train bằng cách văn bản ngành sinh học, rất khó để nó có thể dịch văn bản ngành toán học tốt vì văn phong hai ngành này khác nhau. Đây gọi là bài toán domain adaption, vô cùng khó trong machine learning. Thường trong các bài toán đơn giản, bạn có một khối dữ liệu lớn từ một nguồn, rồi tách ra lấy 8 phần để train, 2 phần để test. Vì thế mà train set và test set sẽ có cùng một nguồn, nói chính xác hơn là cùng một phân bố xác suất. Nhưng mà dù có gần giống nhau như vậy, hai set này cũng vẫn có những khác biệt nhất định. Ta phải đánh đổi giữa sự chi tiết (specificity) và khả năng tổng quát hóa (generalizability) của model. Model càng dự đoán tập train chính xác thì lại càng chi tiết, vì nó phải xem xét giải quyết từng observation một. Có khi một observation không tuân theo quy luật nào cả, model phải đặt ra ngoại lệ, những quy luật mà chỉ đúng với mỗi observation đó hoặc số ít khác. Việc đặt ra quá nhiều ngoại lệ làm giảm khả năng tổng quát hóa của model. Thế nên, để hạn chế những ngoại lệ này, ta chỉ cần model đoán "khá" chính xác trên train set mà thôi. Bù lại model sẽ tổng quát hơn và đoán chính xác hơn trên test set. Suy cho cùng, độ tốt trên test set mới là thứ ta quan tâm sau cùng. 

Thứ hai, khi train model không tối thiểu evaluation function mà tối thiểu **objective function**. Trong trường hợp lý tưởng, hai hàm này trùng nhau. Tuy nhiên, trong đa số trường hợp chúng không giống nhau. Các bạn cảm thấy kì lạ đúng không? Chúng ta dạy model làm tốt một mục tiêu lúc train, nhưng lúc test lại muốn model làm tốt trên một mục tiêu khác. Chúng ta dạy một đằng, nhưng mà lại ra đều một nẻo. Tại sao lại kì lạ như vậy? Lý do là vì evaluation function thường rất khó để tối thiểu hó bằng cách phương pháp toán học (sẽ giải thích ngay sau phần này). Hiểu đơn giản là evaluation function thường có dạng đúng hết thì mới có điểm, cho nên nếu đoán sai thì không biết làm sai để sửa chữa và tiến bộ. Objective function cho chúng ta partial credit, tức là đúng tới đâu cho điểm tới đó, trả lời thế nào cũng có điểm. Vì thế ta có thể tận dụng để thay đổi câu trả lời một chút xem điểm tăng hay giảm, dần dần tìm ra câu trả lời đúng. Objective function được thiết kế có mối quan hệ chặt chẽ với evaluation function, sao cho model tối thiểu hóa objective function cũng tối thiểu hóa evaluation function.

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

