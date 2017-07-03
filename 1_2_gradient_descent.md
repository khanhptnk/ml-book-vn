# Gradient descent (in progress)

Trong hai quá trình của supervised learning, train và test, thì quá trình test đơn giản hơn vì bạn chỉ việc đưa observation vào model, nhận về label dự đoán, và tính giá trị của evaluation function trên test set. Quá trình train phức tạp hơn vì nó phải đảm bảo rằng model phải có khả năng dự đoán tốt trên test set (là dữ liệu không được model nhìn thấy lúc train). Bài viết giải thích tại sao không thể sử dụng các evaluation function như error rate để train model, và giới thiệu về khái niệm objective function. 

### Mục tiêu tối thượng

Khi nói đến việc "giải" một bài toán supervised learning tức là ta đang nói đến việc tìm một phương pháp huấn luyện trên training set sao cho model dự đoán tốt trên test set. Người qua thường ít quan tâm đến độ tốt của model trên training set bởi vì nó thường rất cao. Độ tốt trên training set chỉ thể hiện được sự ghi nhớ của model về những gì đã nhìn thấy. Với một trí thông minh thật sự, ta cần thêm khả năng tổng quát hóa, chính là việc dự đoán tốt trên dữ liệu chưa hề được nhìn thấy.

### Objective function

Để dự đoán tốt nhất trên test set, cách đơn giản nhất là tìm model dự đoán tốt nhất trên training set, và *hy vọng* rằng nó cũng sẽ dự đoán tốt trên test set. Vì thế, ở bài trước ta phát biểu rằng:
1. **Train** (huấn luyện): tìm $$f_w$$ để tối thiểu hóa $$\mathcal{L}_{D_{train}}(f_w)$$.
2. **Test** (kiểm tra): thông báo độ tốt của $$f_w$$ là $$\mathcal{L}_{D_{test}}(f_w)$$.

Tuy nhiên, cách làm này thực sự đang đơn giản hóa vấn đề và thực tế không hiệu quả. Ta sẽ tìm ra **2 vấn đề** trong phát biểu trên và thay đổi để làm nó thực tế và hiệu quả hơn.

Thứ nhất, khi train model ta chỉ muốn tìm ra model dự đoán "khá" chính xác trên training set mà thôi. Vì sao là "khá" chính xác mà không phải là chính xác hoàn toàn? 

Không có điều gì đảm bảo model dự đoán hoàn chính xác trên train set cũng dự đoán tốt trên test set cả. Thậm chí nó có thể dự đoán rất tệ nếu test set rất khác với train set. Điều giống như việc bạn bị "trật tủ" khi đi vậy (ôn một đằng đề ra một kiểu). Thường trong các bài toán, bạn có một khối dữ liệu lớn từ một nguồn nào đó rồi chia ra 80% để train và 20% để test. Vì thế mà train set và test set sẽ có cùng một nguồn, nói chính xác hơn là cùng một phân bố xác suất. Nhưng mà dù có gần giống nhau như vậy, hai set này cũng vẫn có những khác biệt nhất định. Ta phải đánh đổi giữa khả năng ghi nhớ và khả năng tổng quát hóa của model. Model muốn ghi nhớ càng tốt thì lại càng phải xử lý nhiều trường hợp ngoại lệ. Có khi một observation $$x$$ được label là $$y$$ theo một logic rất kì lạ và hiếm gặp, model phải đặt ra ngoại lệ, những quy luật mà chỉ đúng với mỗi observation đó hoặc số ít khác. Việc đặt ra quá nhiều ngoại lệ làm giảm khả năng tổng quát hóa của model vì khả năng ghi nhớ của nó có hạn. Thế nên, để hạn chế những ngoại lệ này, ta chỉ cần model đoán "khá" chính xác trên train set mà thôi. Bù lại model sẽ tổng quát hơn và đoán chính xác hơn trên test set. Suy cho cùng, độ tốt trên test set mới là thứ ta quan tâm sau cùng. 

Thứ hai, trong phát biểu trên ta dùng evaluation function cho cả train và test. Đây là một trường hợp rất lý tưởng và hiếm gặp trong thực tiễn. Trong đa số trường hợp, hàm được model tối ưu lúc train sẽ không phải là hàm dùng để đánh giá nó.  chúng không giống nhau. Các bạn cảm thấy kì lạ đúng không? Tại sao chúng ta "dạy" một đằng, nhưng mà lại "ra đề" một nẻo? Lý do là vì evaluation function thường dùng (như error rate) thường rất khó để tối thiểu hóa bằng cách phương pháp toán học (sẽ giải thích ngay sau phần này). Lý do khái quát là do các evaluation function này thường có dạng:
(a) Đúng hết thì mới có điểm, hoặc 
(b) Tổng của các hàm có dạng như (a). 

Đối với những hàm như vậy, nếu model đoán sai thì không biết sửa chữa theo hướng nào để tiến bộ hơn.

Hàm được model tối ưu lúc train gọi là **objective function** để phân biệt với evaluation function lúc test. Objective function thường cho partial credit, tức là đúng tới đâu cho điểm tới đó và dự đoán thế nào cũng có điểm. Model có thể tận dụng điều này để thay đổi câu trả lời một chút xem điểm tăng hay giảm, dần dần tìm ra câu trả lời đúng. Objective function được thiết kế có mối quan hệ chặt chẽ với evaluation function, sao cho model tối thiểu hóa objective function cũng tối thiểu hóa evaluation function.


### Vì sao không dùng error rate để train model?

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

