from src.task6_lexical_search import build_bm25_index, lexical_search

# Create a test corpus with some sample documents
CORPUS = [
    {
        'content': 'phuong thuc thanh toan shopee bao gom vi dien tu, chuyen khoan ngan hang va thu tien khi nhan hang',
        'metadata': {'source': 'shopeepay.md', 'type': 'payment_method', 'customer_role': 'buyer'}
    },
    {
        'content': 'nguoi ban co the chon phuong thuc thanh toan cod, the tin dung hoac vi dien tu shopee pay',
        'metadata': {'source': 'shopeepay.md', 'type': 'payment_method', 'customer_role': 'seller'}
    },
    {
        'content': 'chinh sach hoan tien ap dung cho tat ca giao dich duoc thuc hien qua shopee pay',
        'metadata': {'source': 'refund_policy.md', 'type': 'refund_policy', 'customer_role': 'buyer'}
    },
    {
        'content': 'de duoc hoan tien, san pham phai con trong tinh trang co the ban duoc va trong vong 7 ngay',
        'metadata': {'source': 'refund_policy.md', 'type': 'refund_policy', 'customer_role': 'buyer'}
    },
    {
        'content': 'chung toi chap nhan tat ca cac loai the visa, mastercard, jcb',
        'metadata': {'source': 'payment_methods.md', 'type': 'payment_method', 'customer_role': 'both'}
    },
    {
        'content': 'khach hang co the su dung vi dien tu shopee de thanh toan nhanh chong va an toan',
        'metadata': {'source': 'payment_methods.md', 'type': 'payment_method', 'customer_role': 'buyer'}
    },
    {
        'content': 'chinh sach bao mat dam bao rang tat ca thong tin ca nhan duoc ma hoa va luu tru an toan',
        'metadata': {'source': 'privacy_policy.md', 'type': 'privacy_policy', 'customer_role': 'both'}
    },
    {
        'content': 'thong tin dang nhap cua ban duoc bao ve bang xac thuc hai yeu to',
        'metadata': {'source': 'privacy_policy.md', 'type': 'privacy_policy', 'customer_role': 'buyer'}
    }
]

# Update the module-level CORPUS
import src.task6_lexical_search as t6
t6.CORPUS = CORPUS
t6.bm25 = t6.build_bm25_index(CORPUS)

# Test the search
print('Test 1: Tim kiem phuong thuc thanh toan shopee')
results = t6.lexical_search('phuong thuc thanh toan shopee', top_k=3)
for r in results:
    print(f'[{r["score"]:.3f}] {r["metadata"]["source"]}: {r["content"][:80]}...')

print()
print('Test 2: Tim kiem hoan tien')
results = t6.lexical_search('hoan tien', top_k=3)
for r in results:
    print(f'[{r["score"]:.3f}] {r["metadata"]["source"]}: {r["content"][:80]}...')

print()
print('Test 3: Tim kiem vi dien tu')
results = t6.lexical_search('vi dien tu', top_k=3)
for r in results:
    print(f'[{r["score"]:.3f}] {r["metadata"]["source"]}: {r["content"][:80]}...')
