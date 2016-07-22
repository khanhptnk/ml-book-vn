# Các khái niệm cơ bản

**Observation** (quan sát): kí hiệu là $$x$$, input trong các bài toán machine learning. Observation thường có dạng một vector $$x = (x_1, x_2, ..., x_d) $$.

**Label** (nhãn): kí hiệu là $$y$$, output của bài toán. Mỗi observation sẽ có một label tương ứng (ví dụ mỗi khuôn mặt sẽ có một cái tên). Label có thể mang nhiều dạng nhưng đều có thể chuyển đổi thành một số thực hoặc một vector. Trong chương này chủ yếu làm việc với label là số thực.

**Model** (mô hình): trong chương này các bạn hiểu là nó là một hàm số $$f(x)$$, nhận vào một observation $$x$$ và trả về một label $$y = f(x)$$. 

**Parameter** (tham số): mọi thứ của model được sử dụng để tính toán ra output. Ví dụ model là một hàm đa thức bậc hai: $$f(x) = ax_1^2 + bx_2 + c$$ thì parameter của nó là bộ ba $$(a, b, c)$$. Tuy nhiên, còn một loại parameter đặc biệt nữa gọi là **hyperparameter**. Hyperparameter là một khái niệm mang tính tương đối và quy ước, thường chỉ các parameter có tính chất hơi mặc định. Đối với hàm đa thức vừa rồi thì bậc của đa thức (bằng 2) có thể được xem là một hyperparameter. Để ngắn gọn, người ta thường gom tất cả parameter của một model lại thành một vector, thường được kí hiệu là $$w$$. Trong ví dụ vừa rồi thì $$w = (a, b, c)$$.  Kí hiệu $$f_w$$ được dùng để chỉ một model đã được xác định tham số. Trong trường hợp cấu trúc model đã được xác định (ví dụ đã biết được nó là đa thức và có bậc 2), thì có thể dùng $$w$$ để chỉ model thay cho $$f_w$$ luôn.