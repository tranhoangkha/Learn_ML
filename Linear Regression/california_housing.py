# Import thư viện
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors=5)
from sklearn.metrics import mean_squared_error, r2_score

# Load dữ liệu California Housing
california = fetch_california_housing()
X = california.data  # Đặc trưng: 8 yếu tố như thu nhập, số phòng, v.v.
y = california.target  # Giá nhà (đơn vị: 100,000 USD)

# Chuyển thành DataFrame để khám phá
df = pd.DataFrame(X, columns=california.feature_names)
print("5 mẫu đầu tiên của dữ liệu:")
print(df.head())
print("5 giá nhà đầu tiên (x100,000 USD):")
print(y[:5])

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình Linear Regression
model = LinearRegression()
# model = KNeighborsRegressor()
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Thử dự đoán một mẫu nhà
sample = X_test[0].reshape(1, -1)  # Lấy mẫu đầu tiên từ tập test
pred_price = model.predict(sample)
print(f"Dự đoán giá nhà cho mẫu: {pred_price[0]:.2f} x 100,000 USD")
print(f"Giá thực tế: {y_test[0]:.2f} x 100,000 USD")