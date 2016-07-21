# Hai góc nhìn về Supervised Learning

### Góc nhìn thứ nhất: ước lượng hàm số

Trò chơi supervised learning: bạn tưởng tượng mình đang chơi một trò chơi với thiên nhiên (là một sức mạnh vô hình điều khiển mọi sự việc của vũ trụ). Đầu tiên, thiên nhiên viết ra một hàm bí ẩn $$f^*$$ nào đó. Sau đó thiên nhiên đưa vào hàm số này một loạt các observation $$x^{(1)}, \cdots, x^{(N)}$$ để tạo ra một loạt các label $$y^{(1)}, \cdots, y^{(N)}$$ tương ứng. Sau đó, thiên nhiên đem giấu hàm $$f^*$$ đi và chỉ chừa lại các cặp observation-label $$(x^{(i)}, y^{(i)})$$ cho chúng ta nhìn thấy. Nhiệm vụ của chúng ta là khôi phục lại được hàm $$f^*$$ bằng một model $$f_w$$ một cách chính xác nhất có thể.

Mọi bài toán supervised learning đều có thể được định nghĩa theo cách này. Ví dụ trong bài toán phân loại văn bản, $$x$$ có thể là một văn bản, $$y$$ là chủ đề của văn bản đó, còn $$f^*$$ là một chuyên gia đọc văn bản và tìm ra chủ đề của chúng.

**Câu hỏi 1**: *giả sử bằng cách nào đó ta biết được rằng $$f^{*}$$ là đa thức bậc 2, tức là $$y = f^{*}(x) = ax_1^2 + bx_2 + c$$, thì cần bao nhiêu cặp $$(x^{(i)}, y^{(i)})$$ để xác định được parameter của $$f^{*}$$?
*

**Trả lời**: *Với mỗi cặp $$(x^{(i)}, y^{(i)})$$ ta xây dựng được một phương trình $$y^{(i)} = f^*(x^{(i)})$$. Vì có 3 ẩn nên ta cần chỉ cần 3 phương trình là giải ra được tham số (nếu tồn tại), tức là cần 3 cặp dữ liệu.* 

Tuy nhiên trong thực tế thì mọi chuyện không đơn giản như vậy. Supervised learning đối mặt với nghịch lý sau đây: vì ta không thể nào giao tiếp được với tự nhiên, nên sẽ không bao giờ biết được $$f^{*}$$ có dạng như thế nào. Vì thế, dù ta có đưa ra một model $$f$$ để ước lượng $$f^{*}$$, cũng không ai biết $$f^{*}$$ là gì để cho biết là ta đang đúng hay sai. Nói cách khác, supervised learning là một trò chơi dự đoán mà không có đáp án đúng. Đọc đến đây các bạn đừng nản lòng mà ngưng đọc tiếp. Tuy là trò chơi này nó có nghe có phần không tưởng, nhưng không riêng gì machine learning, các ngành khoa học cơ bản khác cũng chơi những trò chơi tương tự. Các bạn có nghĩ rằng Einstein nằm mơ thấy chúa thì thầm vào tai mình công thức $$E = mc^2$$?

Cho đến giờ, người ta vẫn phải làm thí nghiệm trong thực tế để kiểm chứng lại các lý thuyết của Einstein cho đến khi nó sai thì thôi. Trong  supervised learning, ta cũng làm một điều tương tự như vậy. Cho dù không biết hình dạng của $$f^{*}$$ ra sao, thì vẫn còn đó các cặp observation-label sinh ra từ hàm này. Ta sẽ đánh giá độ tốt của model $$f_w$$ trên các dữ liệu thực tế này. Ví dụ, nếu nhận được 100 cặp observation-label, ta chỉ dùng khoảng 80 cặp để xây dựng ra $$f_w$$. Còn lại 20 cặp, ta sẽ cho các observation của chúng vào $$f_w$$ để tạo ra các label dự đoán, rồi so sánh chúng với các label thật do $$f^*$$ sinh ra. 80 cặp được dùng để xây dựng ra model gọi là tập huấn luyện (training set), còn 20 cặp dùng để đánh giá model gọi là tập kiểm tra (testing set).



[ML101] accuracy

Câu hỏi 2: tại sao không dùng tất cả dữ liệu để tìm ra $$f_w$$ rồi đánh giá $$f_w$$ trên đó luôn?

Trả lời: trong machine learning, có một nguyên tắc vô cùng, vô cùng quan trọng mà ai cũng phải nhớ: đó là quá trình huấn luyện và kiểm tra phải độc lập với nhau! Có rất nhiều cách để vi phạm nguyên tắc này, và điều dẫn đến một hậu quả "thảm khốc", overfitting. Mình sẽ giải thích về hiện tượng này trong một dịp khác. Nói nôm na là model của bạn sẽ chẳng học được gì khác ngoài những gì nó đã nhìn thấy. Vì thế, bạn phải chia dữ liệu ra thành tập huấn luyện và kiểm tra, và phải làm điều này trước khi huấn luyện và kiểm tra. Tỉ lệ 80:20 ở ví dụ trên là tỉ lệ train:test thường được áp dụng.

Tuy đã biết cách để làm cho trò chơi supervised learning trở nên hợp lệ, nhưng ta vẫn chưa thể chơi được. Có hai vấn đề phát sinh, đó là:

Làm sao để tìm ra được một model tốt từ tập huấn luyện?
Thế nào là một model tốt trên tập kiểm tra?
Để giải quyết hai vấn đề này, ta cần đến góc nhìn thứ hai.

Góc nhìn thứ hai: tối ưu hàm số

Đầu tiên, ta tập trung vào vấn đề thứ hai: giả sử đã tìm được một model $$f_w$$, làm thế nào để thể hiện được độ tốt của nó trên tập kiểm tra bằng một con số cụ thể? Một trong những cách đơn giản nhất đó là đếm xem nó đoán sai bao nhiêu label thật trên tập kiểm tra. Cụ thể hơn, ta giả sử model bị phạt 1 điểm với mỗi lần label dự đoán khác với label thật. Số điểm bị phạt trung bình được gọi là độ sai sót của model (error rate), là một số thực trong đoạn [0, 1]. Theo ngôn ngữ toán học, độ sai sót trên một tập dữ liệu $$D$$ của model $$f_w$$ được tính như sau:

$$ e_D = \frac{1}{|D|} \sum_{(x, y) \in D} \mathbb{I}\{ f_w(x) \neq y \}$$ trong đó:

$$D$$ là một tập dữ liệu gồm các cặp observation-label $$(x, y)$$.
$$|D|$$ là lực lượng của tập dữ liệu (số lượng phần tử).
$$\mathbb{I}\{.\}$$ sẽ trả về 1 nếu logic trong dấu ngoặc nhọn là đúng, 0 nếu sai.
Theo ngôn ngữ lập trình, thì pseudocode sẽ trông giống thế này:

[code language="cpp"]
e = 0
for i = 0 .. N - 1
  if (f(x[i]) != y[i]) e = e + 1
e = e / N
[/code]
Nếu đoán đúng hết tất cả cặp dữ liệu, ta đạt được độ sai sót "trong mơ", 0%. Nhưng nên nhớ đấy là kết quả được đo trên một tập kiểm tra hữu hạn. Kết quả này chỉ đưa ra được một chặn trên và chặn dưới cho kết quả trên tập vô hạn (muốn biết rõ, xem thêm về central limit theorem). Nói cách khác là, nếu tự nhiên có gửi đến một tập dữ liệu mới để đánh giá $$f$$ trên đó, thì chưa chắc ta có thể lặp lại được độ sai sót trên tập cũ. Độ sai sót được đánh giá trên tập dữ liệu càng lớn thì càng đáng tin cậy.

Đến đây chắc các bạn đều hình dung được là nếu độ sai sót càng thấp, thì model càng tốt.

Ta có thể định nghĩa độ tốt theo rất nhiều cách khác nữa. Một cách tổng quát nhất, model sẽ được xác định độ tốt dựa trên một hàm mục tiêu có dạng như sau:

$$ \begin{equation} \mathcal{L}_D(f_w) = \frac{1}{|D|} \sum_{(x, y) \in D} L \left( f_w(x), y \right) \end{equation} $$

Hàm mục tiêu được định nghĩa dựa trên 3 yếu tố:

Tập dữ liệu $$D$$.
Model $$f_w$$.
Hàm $$L$$, gọi là hàm mất mát. Với độ sai sót, thì $$L \left( f_w(x), y \right)$$ chính là  $$\mathbb{I}\{ f(x) \neq y \}$$.
Đến đây, ta đã sẵn sàng để chơi trò chơi supervised learning dưới góc nhìn như một bài toán tối ưu hàm số. Hãy quay lại giải quyết 2 vấn đề trong phần trước:

Làm sao để tìm ra được một model tốt từ tập huấn luyện? --> tìm model cực tiểu hóa giá trị hàm mục tiêu trên tập huấn luyện.
Thế nào là một model tốt trên tập kiểm tra? --> model cho giá trị hàm mục tiêu trên tập kiểm tra càng nhỏ thì càng tốt.
Cụ thể hơn, sau khi định nghĩa được hàm mục tiêu, supervised learning có thể được gói gọn trong 2 bước sau:

1. Tìm $$f_w$$ để tối thiểu hóa $$\mathcal{L}_{D_{train}}(f_w)$$.
2. Thông báo độ tốt của $$f_w$$ là $$\mathcal{L}_{D_{test}}(f_w)$$.

với $$D_{train}$$ và $$D_{test}$$ lần lượt là tập huấn luyện và tập kiểm tra.

