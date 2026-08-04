from src.task6_lexical_search import build_bm25_index, lexical_search

# Create a test corpus with some sample documents
CORPUS = [
    {
        'content': 'phương thức thanh toán shopee bao gồm ví điện tử, chuyển khoản ngân hàng và thu tiền khi nhận hàng',
        'metadata': {'source': 'shopeepay.md', 'type': 'payment_method', 'customer_role': 'buyer'}
    },
    {
        'content': 'người bán có thể chọn phương thức thanh toán cod, thẻ tín dụng hoặc ví điện tử shopee pay',
        'metadata': {'source': 'shopeepay.md', 'type': 'payment_method', 'customer_role': 'seller'}
    },
    {
        'content': 'chính sách hoàn tiền áp dụng cho tất cả giao dịch được thực hiện qua shopee pay',
        'metadata': {'source': 'refund_policy.md', 'type': 'refund_policy', 'customer_role': 'buyer'}
    },
    {
        'content': 'để được hoàn tiền, sản phẩm phải còn trong tình trạng có thể bán được và trong vòng 7 ngày',
        'metadata': {'source': 'refund_policy.md', 'type': 'refund_policy', 'customer_role': 'buyer'}
    },
    {
        'content': 'chúng tôi chấp nhận tất cả các loại thẻ visa, mastercard, jcb',
        'metadata': {'source': 'payment_methods.md', 'type': 'payment_method', 'customer_role': 'both'}
    },
    {
        'content': 'khách hàng có thể sử dụng ví điện tử shopee để thanh toán nhanh chóng và an toàn',
        'metadata': {'source': 'payment_methods.md', 'type': 'payment_method', 'customer_role': 'buyer'}
    },
    {
        'content': 'chính sách bảo mật đảm bảo rằng tất cả thông tin cá nhân được mã hóa và lưu trữ an toàn',
        'metadata': {'source': 'privacy_policy.md', 'type': 'privacy_policy', 'customer_role': 'both'}
    },
    {
        'content': 'thông tin đăng nhập của bạn được bảo vệ bằng xác thực hai yếu tố',
        'metadata': {'source': 'privacy_policy.md', 'type': 'privacy_policy', 'customer_role': 'buyer'}
    }
]

# Update the module-level CORPUS
import src.task6_lexical_search as t6
t6.CORPUS = CORPUS
t6.bm25 = build_bm25_index(CORPUS)

# Test the search
print('Test 1: Tìm kiếm "phương thức thanh toán shopee"')
results = t6.lexical_search('phương thức thanh toán shopee', top_k=3)
for r in results:
    print(f'[{r[\"score\"]:.3f}] {r[\"metadata\"][\"source\"]}: {r[\"content\"][:80]}...')

print('\nTest 2: Tìm kiếm "hoàn tiền"')
results = t6.lexical_search('hoàn tiền', top_k=3)
for r in results:
    print(f'[{r[\"score\"]:.3f}] {r[\"metadata\"][\"source\"]}: {r[\"content\"][:80]}...')

print('\nTest 3: Tìm kiếm "ví điện tử"')
results = t6.lexical_search('ví điện tử', top_k=3)
for r in results:
    print(f'[{r[\"score\"]:.3f}] {r[\"metadata\"][\"source\"]}: {r[\"content\"][:80]}...')
