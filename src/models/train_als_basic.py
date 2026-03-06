import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# ====================================================
# 1. CẤU HÌNH MÔI TRƯỜNG
# ====================================================
os.environ['JAVA_HOME'] = r"C:\Program Files\Java\jre1.8.0_421"
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ====================================================
# 2. KHỞI TẠO SPARK (SAFE MODE 16GB)
# ====================================================
print("🚀 Đang khởi động Spark (Safe Mode cho 16GB RAM)...")
spark = SparkSession.builder \
    .appName("HM_Train_ALS_Basic") \
    .master("local[*]") \
    .config("spark.driver.memory", "5g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ====================================================
# 3. ĐỌC DỮ LIỆU & TẠO CỘT RATING GIẢ
# ====================================================
print("⏳ Đang đọc dữ liệu Train/Val...")
train_path = "./data/processed/transactions_train_set.parquet"
val_path = "./data/processed/transactions_val_set.parquet"

if not os.path.exists(train_path) or not os.path.exists(val_path):
    print("❌ LỖI: Không tìm thấy file dữ liệu!")
    sys.exit()

# Đọc dữ liệu
df_train_raw = spark.read.parquet(train_path)
df_val_raw = spark.read.parquet(val_path)

# --- FIX LỖI Ở ĐÂY ---
# Vì file parquet thiếu cột 'price', ta tạo cột 'rating' mặc định là 1.0
# (Nghĩa là: Cứ có mua thì tính là 1 điểm tương tác)
print("🛠️ Đang thêm cột 'rating' mặc định = 1.0 ...")
df_train = df_train_raw.withColumn("rating", lit(1.0))
df_val = df_val_raw.withColumn("rating", lit(1.0))

# Cache lại
df_train.cache()
df_val.cache()

print(f"✅ Train size: {df_train.count():,}")
print(f"✅ Val size:   {df_val.count():,}")

# ====================================================
# 4. CẤU HÌNH MODEL ALS
# ====================================================
print("🛠️ Đang cấu hình Model ALS...")
als = ALS(
    userCol="user_id_int",
    itemCol="item_id_int",
    ratingCol="rating",       # <--- Đã sửa thành 'rating' thay vì 'price'
    coldStartStrategy="drop",
    nonnegative=True,
    implicitPrefs=True,
    maxIter=5,
    regParam=0.01
)

# ====================================================
# 5. HUẤN LUYỆN (TRAINING)
# ====================================================
print("🏋️‍♂️ Bắt đầu Training (Mời bạn đi pha cà phê)...")
model = als.fit(df_train)
print("🎉 Training xong!")

# ====================================================
# 6. DỰ ĐOÁN THỬ (PREDICTION)
# ====================================================
print("🔮 Đang dự đoán cho tập Validation...")
predictions = model.transform(df_val)
print("📋 5 kết quả dự đoán đầu tiên:")
predictions.show(5)

# ====================================================
# 7. LƯU MODEL
# ====================================================
model_path = "./models/saved/als_basic_model"
model.write().overwrite().save(model_path)
print(f"💾 Đã lưu model vào: {model_path}")

spark.stop()