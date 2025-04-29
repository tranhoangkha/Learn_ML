from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dữ liệu
iris = load_iris()
X = iris.data
y = iris.target

# Xem dữ liệu
# print("5 mẫu đầu tiên của đặc trưng (X):")
# print(X[:5])
# print("Nhãn tương ứng (y):")
# print(y[:5])
# print("Tên loài hoa:")
# print(iris.target_names)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# # Kiểm tra trên tập test
# print("So sánh dự đoán và thực tế trên 5 mẫu đầu tiên của tập test:")
# for i in range(5):
#     print(f"Mẫu {i+1}: Dự đoán = {iris.target_names[y_pred[i]]}, Thực tế = {iris.target_names[y_test[i]]}")

# Test đầu vào tự chọn
sample = [[6.0, 2.7, 5.1, 1.6]]  # Thay đổi giá trị tùy ý
prediction = model.predict(sample)
print(f"Dự đoán loài hoa cho mẫu tự chọn: {iris.target_names[prediction[0]]}")