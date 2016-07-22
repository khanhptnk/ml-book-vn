# Các khái niệm cơ bản

Mình 

**Observation** (quan sát):

**Model** (mô hình): trong bài này các bạn chỉ cần hiểu là nó là một hàm số $$f(x)$$, nhận vào một input $$x$$ và trả về một output $$y = f(x)$$. Ta thường gọi input $$x = (x_1, \cdots, x_d)$$ là observation (lưu ý là đây là một vector nhiều chiều). Output $$y$$ được gọi là label, là thứ mà ta muốn dự đoán.

**Parameter** (tham số): mọi thứ của model được sử dụng để tính toán ra output. Ví dụ model là một hàm đa thức bậc hai: $$f(x) = ax_1^2 + bx_2 + c$$ thì parameter của nó là bộ ba $$(a, b, c)$$. Tuy nhiên, còn một loại parameter đặc biệt nữa gọi là hyperparameter. Hyperparameter là một khái niệm mang tính tương đối và quy ước, thường chỉ các parameter có tính chất hơi mặc định. Đối với hàm đa thức vừa rồi thì bậc của đa thức (bằng 2) có thể được xem là một hyperparameter. Để ngắn gọn, người ta thường gom tất cả parameter của một model lại thành một vector, thường được kí hiệu là $$w$$. Trong ví dụ vừa rồi thì $$w = (a, b, c)$$.  Kí hiệu $$f_w$$ được dùng để chỉ một model đã được xác định tham số. Trong trường hợp cấu trúc model đã được xác định (ví dụ đã biết được nó là đa thức và có bậc 2), thì có thể dùng $$w$$ để chỉ model thay cho $$f_w$$ luôn.