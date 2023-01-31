https://discord.com/channels/992359628979568762/992359845963497512/1069818939200254003  
![](files/tknz-00.jpg)

Chúng ta có thể hardcode vài kênh để có ý nghĩa. Ví dụ:
- kênh 0 cho token có chứa dấu cách (space)
- kênh 1 cho các token được viết hoa ký tự đầu
- kênh 2 cho các ký tự viết hoa toàn bộ
Khi đó:
- Vector nhúng của ` "abc"` là `[0,0,0,x1,x2,..]`
- Vector nhúng của `" abc"` là `[1,0,0,x1,x2,..]`
- Vector nhúng của `" Abc"` là `[1,1,0,x1,x2,..]`
- Vector nhúng của ` "ABC"` là `[0,0,1,x1,x2,..]`
- ...
Như thế cùng một sym gốc "abc" các token sẽ chia xẻ phần lớn vector nhúng (ví dụ độ dài vector nhúng là 1024 mà chỉ dùng 3 kênh đầu để phân biệt sự khác nhau, còn lại 1021 kênh là có giá trị giống hệt nhau). Và chúng ta có thể __nhanh chóng tính được xác xuất đầu ra của mọi biến thể của "abc"__.

Lưu ý là cách trên giả sử p("Xyz") / p("xyz") là không đổi với mọi "xyz" có thể không đúng. __Cách tốt hơn là định nghĩa emb_space, emb_captialize_first, emb_captialize_all là một hàm của emb__.

Hiện tại tokenizer của chúng ta đang mất quá nhiều items vào việc biểu diễn các biến thể của "abc", " abc", "Abc" ... Hơn thế nữa mô hình không thể phát hiện ra sự giống nhau thực sự nếu một vài biến thể là hiếm gặp trong tập dữ liệu huấn luyện.

- - -

Liên hệ tới text tiếng Việt:

I have same kind of thinking when finding a better way to tokenize Vietnamese text.  In Vietnamese following 15 (typo) variants: "ngùoi, nguòi, nguoì, ngừoi, ngưòi, ngưoì, ngùơi, nguời, nguơì, ngừoi, ngưòi, ngưoì, ngừơi, người, ngươì", are belongs to correct word "người". If we use English-alike BPE there will be a lot of uncessesary tokens that model need to handle since those typos are common in Vietnamese text corpus (in chatting and commenting where people type very fast and no need to be 100% correct). Peng idea remind me that we should consider tokenization seriously, since it done, we cannot have change to fine-tune or fix it. 

Quay trở lại ví dụ trên, dùng symato, sym "nguoi" xuất hiện 15 lần, nghĩa là nó được tập trung biểu diễn, còn các biến thể của nó cũng chỉ gom lại còn là 02 marktones: là f, và wf. Như vậy số lượng tokens giảm đi đáng kể, tập trung biểu diễn cho sym "nguoi" => giúp mô hình chịu lỗi cao hơn và tiết kiệm tài nguyên.

