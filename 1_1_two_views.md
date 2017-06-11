# Hai góc nhìn về Supervised Learning

Tiếp cận supervised learning thực chất không đòi hỏi quá nhiều kiến thức cao siêu. Nó có thể được quy về thành một bài toán tối thiểu hàm số cơ bản trong toán học.

### Góc nhìn thứ nhất: ước lượng hàm số

Bạn hãy tưởng tượng rằng đang chơi một trò chơi với thiên nhiên \(là một sức mạnh vô hình điều khiển mọi sự việc của vũ trụ\). Đầu tiên, thiên nhiên viết ra một hàm bí ẩn $$f^*$$ nào đó. Sau đó thiên nhiên đưa vào hàm số này một loạt các observation $$x^{(1)}, \cdots, x^{(N)}$$ để tạo ra một loạt các label $$y^{(1)}, \cdots, y^{(N)}$$ tương ứng. Sau đó, thiên nhiên đem giấu hàm $$f^*$$ đi và chỉ chừa lại các cặp observation-label $$(x^{(i)}, y^{(i)})$$ cho chúng ta nhìn thấy. Nhiệm vụ của chúng ta là khôi phục lại được hàm $$f^*$$ bằng một model $$f_w$$ một cách chính xác nhất có thể. Ta gọi đây là **trò chơi supervised learning**.

Mọi bài toán supervised learning đều có thể được định nghĩa theo cách này. Ví dụ trong bài toán phân loại văn bản, observation $$x$$ có thể là một văn bản, label $$y$$ là chủ đề của văn bản đó, còn $$f^*$$ là một chuyên gia đọc văn bản và tìm ra chủ đề của chúng.

**Q1**: _giả sử bằng cách nào đó ta biết được rằng _$$x = (x_1, x_2)$$_ \(vector 2 chiều\) và _$$y = f^*(x) = ax_1^2 + bx_2 + c$$. Cần bao nhiêu điểm dữ liệu $$(x, y)$$ để có thể tìm ra parameter$$(a, b, c)$$?

**A1**: _Với mỗi cặp _$$(x, y)$$_ ta xây dựng được một phương trình _$$$$$$y = ax_1^2 + bx_2 + c$$. Vì có 3 ẩn nên ta cần chỉ cần 3 phương trình là giải ra được tham số \(nếu tồn tại\). Tức là cần 3 điểm dữ liệu. 

Tuy nhiên trong thực tế thì mọi chuyện không đơn giản như vậy. Supervised learning đối mặt với nghịch lý sau đây: vì ta không thể nào giao tiếp được với tự nhiên, nên sẽ không bao giờ biết được $$f^{*}$$ có dạng như thế nào. Vì thế, dù ta có đưa ra một model $$f$$ để ước lượng $$f^{*}$$, cũng không ai biết $$f^{*}$$ là gì để cho biết là ta đang đúng hay sai. Nói cách khác, supervised learning là một trò chơi dự đoán mà không có đáp án đúng.

Đọc đến đây các bạn đừng nản lòng. Tuy là trò chơi này nghe có phần không tưởng, nhưng không riêng gì machine learning, các ngành khoa học cơ bản khác cũng chơi những trò chơi tương tự. Các bạn có nghĩ rằng Einstein nằm mơ thấy thần linh thì thầm vào tai mình công thức $$E = mc^2$$?

Cho đến giờ, người ta vẫn phải làm thí nghiệm trong thực tế để kiểm chứng lại các lý thuyết của Einstein cho đến khi nó sai thì thôi. Trong  supervised learning, ta cũng làm một điều tương tự như vậy. Cho dù không biết hình dạng của $$f^{*}$$ ra sao, thì vẫn còn đó các cặp observation-label sinh ra từ hàm này. Ta sẽ đánh giá độ tốt của model $$f_w$$ trên các dữ liệu thực tế này. Ví dụ, nếu nhận được 100 cặp observation-label, ta chỉ dùng khoảng 80 cặp để xây dựng ra $$f_w$$. Còn lại 20 cặp, ta sẽ cho các observation của chúng vào $$f_w$$ để tạo ra các label dự đoán, rồi so sánh chúng với các label thật do $$f^*$$ sinh ra. 80 cặp được dùng để xây dựng ra model gọi là tập huấn luyện \(training set\), còn 20 cặp dùng để đánh giá model gọi là tập kiểm tra \(testing set\).

![](http://khanhxnguyen.com/wp-content/uploads/2016/05/ML101-accuracy.png)

**Q2**: _Tại sao không dùng tất cả dữ liệu để tìm ra _$$f_w$$_ rồi đánh giá _$$f_w$$_ trên đó luôn?_

**A2**: _Trong machine learning, có một nguyên tắc vô cùng, vô cùng quan trọng mà ai cũng phải nhớ: đó là quá trình huấn luyện và kiểm tra phải độc lập với nhau! Có rất nhiều cách để vi phạm nguyên tắc này, và điều dẫn đến một hậu quả "thảm khốc", overfitting. Mình sẽ giải thích về hiện tượng này trong một dịp khác. Nói nôm na là model của bạn sẽ chẳng học được gì khác ngoài những gì nó đã nhìn thấy. Vì thế, bạn phải chia dữ liệu ra thành tập huấn luyện và kiểm tra, và phải làm điều này trước khi huấn luyện và kiểm tra. Tỉ lệ 80:20 ở ví dụ trên là tỉ lệ train:test thường được áp dụng._

Tuy đã biết cách để làm cho trò chơi supervised learning trở nên hợp lệ, nhưng ta vẫn chưa thể chơi được. Có hai vấn đề phát sinh, đó là:

1. Làm sao để tìm ra được một model tốt từ tập huấn luyện?
2. Thế nào là một model tốt trên tập kiểm tra?

Để giải quyết hai vấn đề này, ta cần đến góc nhìn thứ hai.

### Góc nhìn thứ hai: tối ưu hàm số

Đầu tiên, ta tập trung vào vấn đề thứ hai: giả sử đã tìm được một model $$f_w$$, làm thế nào để thể hiện được độ tốt của nó trên tập kiểm tra bằng một con số cụ thể? Một trong những cách đơn giản nhất đó là đếm xem nó đoán sai bao nhiêu label thật trên tập kiểm tra. Cụ thể hơn, ta giả sử model bị phạt 1 điểm với mỗi lần label dự đoán khác với label thật. Số điểm bị phạt trung bình được gọi là độ sai sót của model \(error rate\), là một số thực trong đoạn \[0, 1\]. Theo ngôn ngữ toán học, độ sai sót trên một tập dữ liệu $$D$$ của model $$f_w$$ được tính như sau:


$$
e_D = \frac{1}{|D|} \sum_{(x, y) \in D} \mathbb{I}\{ f_w(x) \neq y \}
$$


trong đó:

* $$D$$ là một tập dữ liệu gồm các cặp observation-label $$(x, y)$$.
* $$|D|$$ là lực lượng của tập dữ liệu \(số lượng phần tử\).
* $$\mathbb{I}\{.\}$$ sẽ trả về 1 nếu logic trong dấu ngoặc nhọn là đúng, 0 nếu sai.

Theo ngôn ngữ lập trình, thì pseudocode sẽ trông giống thế này:

```
e = 0
for i = 0 .. N - 1
  if (f(x[i]) != y[i]) e = e + 1
e = e / N
```

Nếu đoán đúng hết tất cả cặp dữ liệu, ta đạt được độ sai sót "trong mơ", 0%. Nhưng nên nhớ đấy là kết quả được đo trên một tập kiểm tra hữu hạn. Kết quả này chỉ đưa ra được một chặn trên và chặn dưới cho kết quả trên tập vô hạn \(muốn biết rõ, xem thêm về [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)\). Nói cách khác, nếu tự nhiên có gửi đến một tập dữ liệu mới để đánh giá $$f_w$$ trên đó, thì chưa chắc ta có thể lặp lại được độ sai sót trên tập cũ. Độ sai sót được đánh giá trên tập dữ liệu càng lớn thì càng đáng tin cậy.

Đến đây chắc các bạn đều hình dung được là nếu **độ sai sót càng thấp, thì model càng tốt**.

Ta có thể định nghĩa độ tốt theo rất nhiều cách khác nữa. Một cách tổng quát, model sẽ được xác định độ tốt dựa trên một hàm mục tiêu có dạng như sau:


$$
\mathcal{L}_D(f_w) = \frac{1}{|D|} \sum_{(x, y) \in D} L \left( f_w(x), y \right)
$$


Hàm mục tiêu được định nghĩa dựa trên 3 yếu tố:

1. Tập dữ liệu $$D$$.
2. Model $$f_w$$.
3. Hàm $$L$$, gọi là **hàm mất mát**. Với độ sai sót, thì $$L \left( f_w(x), y \right)$$ chính là  $$\mathbb{I}\{ f(x) \neq y \}$$.

Đến đây, ta đã sẵn sàng để chơi trò chơi supervised learning dưới góc nhìn như một bài toán tối ưu hàm số. Hãy quay lại giải quyết 2 vấn đề trong phần trước:

1. Làm sao để tìm ra được một model tốt từ tập huấn luyện? $$\rightarrow$$ tìm model cực tiểu hóa giá trị hàm mục tiêu trên tập huấn luyện.
2. Thế nào là một model tốt trên tập kiểm tra? $$\rightarrow$$ model cho giá trị hàm mục tiêu trên tập kiểm tra càng nhỏ thì càng tốt.

Cụ thể hơn, sau khi định nghĩa được hàm mục tiêu, supervised learning có thể được gói gọn trong 2 bước sau:

1. **Huấn luyện**: tìm $$f_w$$ để tối thiểu hóa $$\mathcal{L}_{D_{train}}(f_w)$$.
2. **Kiểm tra**: thông báo độ tốt của $$f_w$$ là $$\mathcal{L}_{D_{test}}(f_w)$$.

với $$D_{train}$$ và $$D_{test}$$ lần lượt là tập huấn luyện và tập kiểm tra.

