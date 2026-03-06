import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, collect_list

# ====================================================
# 1. CẤU HÌNH (Vẫn giữ cấu hình 16GB an toàn)
# ====================================================
os.environ['JAVA_HOME'] = r"C:\Program Files\Java\jre1.8.0_421"
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

print("🚀 Đang khởi động Spark để chấm điểm...")
spark = SparkSession.builder \
    .appName("HM_Evaluate_MAP12") \
    .master("local[*]") \
    .config("spark.driver.memory", "5g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ====================================================
# 2. LOAD MODEL VÀ DỮ LIỆU
# ====================================================
model_path = "./models/saved/als_basic_model"
val_path = "./data/processed/transactions_val_set.parquet"

if not os.path.exists(model_path) or not os.path.exists(val_path):
    print("❌ LỖI: Không tìm thấy Model hoặc file Validation!")
    print("👉 Hãy chạy file train_als_basic.py trước đã.")
    sys.exit()

print(f"📥 Đang load model từ: {model_path}")
model = ALSModel.load(model_path)

print("⏳ Đang đọc tập Validation...")
df_val = spark.read.parquet(val_path)
# Tạo cột rating giả (nếu chưa có) để tránh lỗi
from pyspark.sql.functions import lit
if "rating" not in df_val.columns:
    df_val = df_val.withColumn("rating", lit(1.0))

# ====================================================
# 3. CHUẨN BỊ "ĐÁP ÁN THẬT" (GROUND TRUTH)
# ====================================================
print("DATA PREP: Gom nhóm các món hàng User thực tế đã mua...")
# Gom tất cả item_id mà user đã mua thành 1 list. Ví dụ: User A -> [Item 1, Item 5]
actual_items = df_val.groupBy("user_id_int") \
    .agg(collect_list("item_id_int").alias("truth"))

# Chỉ lấy danh sách các User có trong tập Validation để dự đoán (Tiết kiệm RAM)
users_in_val = df_val.select("user_id_int").distinct()

# ====================================================
# 4. SINH RA "DỰ ĐOÁN" (PREDICTIONS)
# ====================================================
print("PREDICTION: Đang gợi ý Top 12 sản phẩm cho mỗi User (recommendForUserSubset)...")
# Hàm này cực mạnh: Nó tự tìm 12 món có điểm cao nhất cho từng user
recs = model.recommendForUserSubset(users_in_val, 12)

# ====================================================
# 5. TÍNH TOÁN MAP@12
# ====================================================
print("EVALUATION: Đang so sánh kết quả...")

# Join bảng Gợi ý (recs) với bảng Thực tế (actual_items)
# Kết quả sẽ là: UserID | [List Gợi ý] | [List Thực tế]
joined_data = recs.join(actual_items, "user_id_int")

# Chuyển đổi format để đưa vào RankingMetrics
# RankingMetrics cần RDD dạng: (prediction_list, truth_list)
prediction_and_labels = joined_data.rdd.map(lambda row: (
    [x.item_id_int for x in row.recommendations], # Lấy list ID gợi ý
    row.truth                                     # Lấy list ID thật
))

metrics = RankingMetrics(prediction_and_labels)

# Lấy chỉ số MAP (Mean Average Precision)
map_score = metrics.meanAveragePrecision
ndcg_score = metrics.ndcgAt(12) # Normalized Discounted Cumulative Gain

print("\n" + "="*40)
print(f"📊 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Trên tập Validation)")
print("="*40)
print(f"🎯 MAP (Mean Average Precision): {map_score:.5f}")
print(f"🌟 NDCG@12:                     {ndcg_score:.5f}")
print("="*40)

print("\n🔍 GIẢI THÍCH NHANH:")
print("- MAP càng cao càng tốt (Max = 1.0).")
print("- Với bài toán H&M cực khó này, MAP thường rất thấp (0.01 - 0.03).")
print("- Nếu > 0.0, nghĩa là mô hình ĐANG HỌC ĐƯỢC CÁI GÌ ĐÓ (hơn là đoán mò).")

spark.stop()