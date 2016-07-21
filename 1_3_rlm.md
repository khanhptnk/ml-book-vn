# Regularized Risk Minimization 

Trong phần này, chúng ta sẽ hoàn tất những hiểu biết về overfitting và đưa ra một thuật toán supervised learning hiệu quả hơn ERM để chống lại overfitting. Nhưng trước khi đó ta cùng ôn lại những gì đã học ở phần trước bằng một số câu hỏi ngắn như sau:

**Q1** *: Overfitting là gì?*

**A1** *: Là khi model không có khả năng tổng quát từ những gì đã học được: độ sai sót trên tập huấn luyện nhỏ, trên tập kiểm tra to.*

**Q2** *: Tại sao overfitting lại có hại?*

**A2** *: Vì dữ liệu lúc nào cũng chứa noise, mà model bị overfitting rất nhạy cảm với noise. Noise làm cho model tìm được phức tạp quá mức cần thiết.*

**Q3** *: Làm sao để biết được model có bị overfitting hay không?*

**A3** *: Theo dõi learning curve.*

**Q4** *: Làm sao để không bị overfitting?*

**A4** *: Đây không phải là một câu hỏi đúng vì overfitting là một khái niệm tương đối, tùy theo "cảm giác" của bạn. Nếu bạn đang nói về chuyện làm sao để $$\mathcal{L}_{D_{train}}$$ trùng với $$\mathcal{L}_{\mathcal{D}}$$ thì câu trả lời là không thể, trừ phi có vô hạn dữ liệu.*

Chúng ta sẽ tìm hiểu sâu hơn về nguyên nhân gây ra overfitting. Như chúng ta đã biết, noise không phải là nguyên nhân trực tiếp gây ra overfitting. Vậy những yếu tố nào gây ra overfitting?

### Nguyên nhân gây ra overfitting

Overfitting là sản phẩm của sự cộng hưởng giữa các yếu tố sau:

1. **Áp dụng ERM**.

2. **Giới hạn về dữ liệu**: khi có thêm các cặp observation-label, hiển nhiên ta có thêm thông tin về mối quan hệ giữa chúng. Cụ thể hơn, ta thấy rằng $$\mathcal{L}_{D_{train}}$$ sẽ hội tụ về $$\mathcal{L}_{\mathcal{D}}$$ khi độ lớn của $$D_{train}$$ tiến đến vô cực. Khi hai đại lượng này trùng nhau thì overfitting hoàn toàn biến mất (theo định nghĩa). Vì thế, càng có nhiều dữ liệu huấn luyện thì càng ít bị overfitting.

3. **Tập model quá "mạnh"**: khi chọn một dạng model $$f_w$$ và thay đổi parameter $$w$$ và các hyperparameter khác, ta được một **tập model** (model family). Ví dụ nếu $$f_w$$ là một đa thức bậc một, thì tập model là tập hợp tất cả các đa thức bậc một (có dạng $$y = f_w(x) = w_1x + w_2$$). Dù có vô số đa thức như vậy, nhưng mà đây được xem như một tập model "yếu" bởi vì nó không biểu diễu được các hàm phi tuyến tính. Vì bản chất machine learning là ước lượng hàm số, sử dụng một tập model mạnh hơn, thậm chí có khả năng mô phỏng tất cả dạng hàm số tưởng chừng như là một ý hay. Nhưng thực tế đây lại là một ý tưởng này rất tồi. Vì sao?

### Vì sao dùngtập model quá mạnh lại không tốt?

Giả sử có một cuộc thi trong đó ta yêu cầu mỗi thí sinh phải vẽ được một đường đi qua nhiều nhất các điểm cho trước. Thí sinh tham dự có 2 người: một người là họa sĩ, anh ta rất khéo tay và có thể vẽ tất cả các loại đường cong thẳng; người còn lại là một anh chàng vụng về với cây thước kẻ, anh ta chỉ có thể vẽ đường thẳng. Dĩ nhiên là anh họa sĩ sẽ thắng trong trò chơi này.

Nhưng hãy phân tích phản xạ của hai thí sinh trong tình huống sau đây: ta cho đề bài ban đầu là các điểm trên một đường thẳng; sau khi hai người vẽ xong, ta chỉ dịch chuyển một điểm lệch ra khỏi đường thẳng một đoạn nhỏ. Hiển nhiên là ban đầu cả hai người đều vẽ được một đường thẳng đi qua tất cả các điểm. Nhưng sau khi một điểm bị dịch chuyển, anh họa sĩ sẽ vẽ ra một đường hoàn toàn khác với đường thẳng ban đầu, cho dù điểm đó chỉ bị xê dịch chút đỉnh. Ngược lại, anh vụng về thì sẽ vẫn giữ nguyên đáp áp vì đó là đường tốt nhất anh có thể vẽ. Điều ta thấy được ở đây đó là anh họa sĩ, vì quá tài hoa, nên anh rất nhạy cảm với những thay đổi nhỏ trong các điểm dữ liệu. Còn anh vụng về, vì năng lực của anh có hạn, nên thường anh sẽ ít bị ảnh hưởng hơn.

Nếu như đây không phải là một cuộc thi vẽ qua nhiều điểm mà là một bài toán machine learning, có lẽ anh họa sĩ đã thua rồi. Bởi vì điểm bị dịch chuyển có thể là do tác động của noise để hòng đánh lừa anh. Anh họa sĩ đại diện cho một tập model cực mạnh, có khả năng mô phỏng mọi hàm số. Một tập model mạnh như vậy rất nhạy cảm với noise và dễ dàng bị overfitting.

![](http://khanhxnguyen.com/wp-content/uploads/2016/06/Model-quá-mạnh.png)

### Sự kết hợp giữa các yếu tố gây overfitting 

Các yếu tố gây ra overfitting phải phối hợp với nhau thì mới đủ điều kiện cho nó xuất hiện. Ta xem xét hai tình huống thường gặp sau:

1. Có nhiều dữ liệu: ta có thể vô tư dùng ERM, tập model mạnh mà không lo về overfitting. Đây chính là lý do mà thế giới hân hoan khi Big Data xuất hiện.

2. Làm việc với tập model yếu: các model thường ít bị overfitting mà bị một hội chứng chị em ngược lại với nó, gọi là **underfitting**. Đây là khi model quá đơn giản so với quan hệ cần tìm. Lúc này, dù có tăng thêm dữ liệu cũng không giúp cho model chính xác thêm. Điều cần làm đó là tăng sức mạnh (tăng số lượng tham số hoặc thay đổi dạng) của tập model.

Mình cũng xin dành ra vài dòng để nói về hiện tượng "cuồng" deep learning và áp dụng deep learning lên mọi bài toán. Các model của deep learning là các model mạng neuron cực mạnh nên cần rất nhiều dữ liệu để không bị overfitting. Đó là lý do mà dù các model deep learning này không mới, thậm chí là những model đầu tiên của machine learning, nhưng phải chờ đến kỷ nguyên Big Data hiện tại chúng mới phát huy sức mạnh. Nếu không am hiểu về overfitting và áp dụng deep learning vô tội vạ lên những tập dữ liệu chỉ có vài trăm cặp dữ liệu thì thường đạt đượt kết quả không cao. Khi gặp những điều kiện dữ liệu eo hẹp như vậy, nên bắt đầu từ những model đơn giản hơn trước.  Trong machine learning có một định lý gọi là "no free lunch" nói rằng không có một model nào tốt nhất cho tất cả các loại dữ liệu. Vì thế, tùy vào bài toán, vào tính chất và số lượng dữ liệu sẵn có, ta mới xác định được model phù hợp.

### Regularized risk minimization

Trong bài trước, ta đã biết được một phương pháp để giảm thiểu overfitting, **early stopping**. Ba yếu tố gây ra overfitting cũng gợi ý cho chúng ta những cách khác để khắc phục vấn đề này. Trong đó, yếu tố thứ hai đưa ra giải pháp đơn giản nhất: tăng kích thước tập huấn luyện. Sau đây, mình sẽ giới thiệu một phương pháp nhằm loại trừ đi yếu tố thứ nhất và thứ ba, được gọi là phương pháp **bình thường hóa tham số** (regularization). Phương pháp này sẽ thay đổi hàm mục tiêu của ERM nhằm hạn chế sức mạnh của model.

Giả sử rằng đã lỡ chọn một tập model quá mạnh. Thì không cần phải thay đổi dạng model, ta vẫn có thể hạn chế sức mạnh của nó đi bằng cách giới hạn không gian tham số (parameter space/domain) của model. Xét hai tập model $$A = \{ f_w : w \in X\}$$ và $$B = \{ f_{w'} : w' \in Y\}$$ (ký hiệu $$S = \{s : c\}$$ đọc là "tập $$S$$ gồm các phần tử $$s$$ sao cho điều kiện $$c$$ thỏa mãn). $$X$$ hoặc $$Y$$ được gọi là không gian tham số của tập model tương ứng. Trong trường hợp này, nếu $$X \subset Y$$ thì rõ ràng tập model $$B$$ biểu diễn được mọi hàm số tập model $$A$$ biểu diễn được, tức là $$B$$ mạnh hơn $$A$$.

Nếu $$w$$ là một vector số thực có $$d$$ thành phần, tập hợp các giá trị $$w$$ có thể nhận, hay không gian tham số của $$w$$, là $$\mathbb{R}^d$$. Trong không gian này, mỗi thành phần của $$w$$ đều được tự do bay nhảy trong khoảng $$(-\infty,\infty)$$. Muốn thu nhỏ lại không gian này, ta cần một cơ chế để không cho các thành phần của $$w$$ trở nên quá lớn.

Để làm được điều đó, ý tưởng ở đây là định nghĩa một đại lượng để khái quát được "độ lớn" của vector $$w$$ và cố gắng giảm thiểu nó. Ta ký hiệu đại lượng này là, $$R(w)$$, là một hàm số phụ thuộc vào $$w$$. Khi huấn luyện để tìm ra $$w$$, cùng với việc giảm thiểu $$R(w)$$, ta cũng nhớ là mình vẫn cần phải giảm thiểu hàm mục tiêu của ERM, $$\mathcal{L}_{D_{train}}(w)$$. Để thể hiện được việc phải giảm thiểu cùng một lúc hai hàm số, ta sẽ giảm thiểu tổng của chúng. Cụ thể, ta định nghĩa lại mục tiêu ở bước (1) của supervised learning như sau:

$$
 w = \arg\min_{w'} \mathcal{L}_{D_{train}}(w') + \lambda R(w')
$$

Quy tắc này được gọi là **regularized loss minimization** (RLM), một mở mộng của ERM. Chú ý là đối với hàm mục tiêu của RLM, không nhất thiết là $$\mathcal{L}_{D_{train}}$$ phải đạt giá trị tối thiểu để cho tổng $$\mathcal{L}_{D_{train}} + \lambda R$$ trở nên tối thiểu. Nếu một model tối thiểu hóa $$\mathcal{L}_{D_{train}}$$ nhưng lại làm cho $$R$$ đạt giá trị lớn thì vẫn có cơ hội để chọn một model khác, dù có $$\mathcal{L}_{D_{train}}$$ lớn hơn nhưng lại cho giá trị của $$R$$ nhỏ hơn nhiều. Nói cách khác, ta có thể lựa chọn được một model đơn giản, dù nó không dự đoán hoàn hảo tập huấn luyện. Hàm mục tiêu của RLM đang đưa model đi gần đến Occam's razor hết mức có thể. Ta chấp nhận hy sinh độ chính xác trên tập huấn luyện để giảm độ phức tạp của model, miễn là giảm được hàm mục tiêu tổng. Tuy nhiên, đây là sự đánh đổi hoàn toàn có lợi cho ta.

Hằng số $$\lambda$$ trong hàm mục tiêu được gọi là **hằng số bình thường hóa**, là một hyperparameter của model. Sự xuất hiện của $$\lambda$$ trong hàm mục tiêu làm cho vai trò của $$\mathcal{L}_{D_{train}}$$ và $$R$$ trở nên *bất đối xứng*: nếu ta tăng $$\mathcal{L}_{D_{train}}$$ lên $$1$$ đơn vị thì hàm mục tiêu tăng lên $$1$$ đơn vị; trong khi đó nếu tăng $$R$$ lên $$1$$ đơn vị thì hàm mục tiêu tăng lên thêm $$\lambda$$ đơn vị. Tức là $$1$$ đơn vị của $$\mathcal{L}_{D_{train}}$$ có giá trị bằng $$1 / \lambda$$ đơn vị của $$R$$. Thông thường, ta thường đặt $$\lambda$$ rất nhỏ, ví dụ $$\lambda = 10^{-6}$$. Lúc này, $$1$$ đơn vị của $$\mathcal{L}_{D_{train}}$$ bằng đến $$10^6$$ đơn vị của $$R$$. Điều này thể hiện rằng ta muốn ưu tiên vào tối thiểu hóa $$\mathcal{L}_{D_{train}}$$ hơn là $$R$$.

### Các hàm bình thường hóa thường gặp

$$R(w)$$ thường gặp nhất là norm của vector. Có rất nhiều loại norm, mình sẽ giới thiệu hai loại norm phổ biến nhất.

**1-norm**: $$ R(w) = ||w||_1 = \sum_{i = 1}^d |w_i|$$ 

tức là tổng của trị tuyệt đối của các thành phần. 1-norm đặc biệt ở chỗ là, khi đưa vào hàm mục tiêu, nó sẽ thường cho ra model thưa, tức là một vector $$w$$ có nhiều thành phần bằng 0. Model thưa rất có lợi thế trong tính toán và lưu trữ vì ta có thể phớt lờ đi các thành phần bằng 0.

**squared 2-norm**: $$ R(w) = ||w||_2^2 = \sum_{i = 1}^d w_i^2$$ 

chính là bình phương độ dài của vector $$w$$. Sở dĩ ta phải bình phương là để giúp cho việc tính đạo hàm được dễ hơn khi tối ưu hàm mục tiêu. Mình sẽ nói kỹ hơn về vấn đề này vào dịp khác.

