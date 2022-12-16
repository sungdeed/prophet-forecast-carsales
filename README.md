# prophet-forecast-carsales

## ในบทความนี้เราจะมาทำการใช้ prophet มาทำนายยอดขายรถยนต์โดยใช้ sentimental analysis โดยเริ่มแรกเราจะทำการติดตั้งแพคเกจสำหรับ Prophet
```
pip install prophet
```
### โหลด library ที่จำเป็น
```
import pandas as pd
from prophet import Prophet
import numpy as np
```
Pandas ใช้ในการจัดการข้อมูลที่อยู่ในรูป csv
Numpy เพื่อใช้ประมวลผล
Prophet เพื่อใช้โมเดลในการทำนายผล

### import data 
```
df = pd.read_csv('Sale.csv')
df
```
โหลดข้อมูลได้ทำการที่เก็บมา โดยข้อมูลแต่ละ Feature คือ
Altis = Toyota Altis  
Civic = Honda Civic
Yaris = Toyota Yaris
City = Honda City 
Date = วันที่่โดยเราเริ่มเก็บวันที่ 01-01-2018 ถึง  01-10-2022
Pos = positive ความคิดเห็นเชิงบวก
Neg = negative ความคิดเห็นแง่ลบ

ตัวอย่าง 
ข้อมูลในแถวที่ 0 จะสรุปได้ว่า

ปี 2018-01-01 Altis มียอดขายรถยนต์อยู่ที่ 1,639 คัน มีความคิดเห็นเชิงบวกอยู่ที่ 10 ครั้ง(Altis pos = 10) ความคิดเห็นเชิงลบ 10 ครั้ง (Altis neg = 10)

ปี 2018-01-01 Civic  มียอดขายรถยนต์อยู่ที่ 2,133 คัน มีความคิดเห็นเชิงบวกอยู่ที่ 14  ครั้ง(Civic pos = 10) ความคิดเห็นเชิงลบ 4 ครั้ง (Civic neg = 10)

ปี 2018-01-01 Yaris  มียอดขายรถยนต์อยู่ที่ 4,274 คัน มีความคิดเห็นเชิงบวกอยู่ที่ 18  ครั้ง(Yaris pos = 18) ความคิดเห็นเชิงลบ 2 ครั้ง (Yaris neg = 2)

ปี 2018-01-01 City  มียอดขายรถยนต์อยู่ที่ 4,239 คัน มีความคิดเห็นเชิงบวกอยู่ที่ 9  ครั้ง(City pos = 9) ความคิดเห็นเชิงลบ 5 ครั้ง (City neg = 5)


### Data preparation
```
altis = df[['Date', 'Altis']]
altis.dropna(inplace = True)
altis.columns = ['ds', 'y']
altis
```
โมเดล Prophet ถูกกำหนดไว้ว่าคคอลล์ลัมข้อมูลที่จะนำไปคำนวน Date = ds,  input data = y

กำหนดให้ Date = ds 

รุ่นรถยนต์ที่ต้องการทำนาย Altis = y

ลบข้อมูล Null

### train model
```
model = Prophet()
model.fit(altis)
```
สร้างฟังก์ชั่น Model เพื่อเรียกใช้ library Prophet  

Model.fit(altis) เพื่อให้ข้อโมเดลมาเรียนรู้ข้อมูล
### add Regressor
```
model_new.add_regressor('Altis(pos)')
model_new.add_regressor('Altis(neg)')
```
### Forecat
```
future = model.make_future_dataframe(periods=12, freq = 'M')
future.tail(12)
```
สร้างฟังก์ชั่น Futureไว้ล่วงหน้าเพื่อให้อัลกอริทึมทำนาย โดยกำหนดให้โมเดลทำนาย 12 เดือนล่วงหน้า โดยเริ่มทำนายวันที่ 31-10-2022 ถึงวันที่ 30-09-2-23
```
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```
Forecast = ให้โมเดลทำนายยอดขายรถยนต์ 12 ปีล่วงหน้า 

Yhat = ค่าที่โมเดลทำนายออกมา เช่นในวันที่ 30-09-2023 โมเดลของราได้ทำนายออกมาว่ารถรุ่น Altis จะมียอดขายอยู่ที่ประมาณ 911 คัน และจะอยู่ในช่วงประมาณ 540-1,279 คัน ซึ่งก็คือค่าในช่วง  yhat_lower ถึง yhat_upper

โดยโมเดลได้ทำนายค่าออกมาเป็นช่วง yhat_lower ถึง yhat_upper ในส่วนนี้โมเดลก็ได้เอาไปใช้ในการหาแนวโน้มยอดขาย
### Plot model
```
fig1 = model.plot(forecast)
```
ให้โมเดลพลอตกราฟที่ทำนายไว้โดยช่วงที่มี Data point คือ ช่วงทีโมเดลเรียนรู้จากข้อมูล และเส้นหลังจากนั้นคือ เส้นที่โมเดลได้ทำนาย โดยผลลัพธ์ที่โมเดลทำนายออกมาคือ ยอดขายอีก 12 เดือนข้างหน้ามีการเปลี่ยนยอดขายอย่างมาก และยอดขายสูงสุดที่ทำได้คือประมาณ 1,186 คัน และยอดขายต่ำสุดคือ 436 คัน 
```
fig2 = model.plot_components(forecast)
```
ให้โมเดลทำนายแนวโน้มยอดขายรถยนต์ Altis โดยโมเดลทำนายออกมาว่ายอดขายรถยนต์ Toyota Altis มีแนวโน้มลดลงอย่างมาก เห็นได้จากยอดขายที่มีแนวโน้มลดลง
### Evaluate
```
import pandas as pd
cutoffs = pd.date_range(start='2019-01-01', end='2020-06-01', freq='2MS')
print(cutoffs)
```
```
from prophet.diagnostics import cross_validation
df_cv = cross_validation(model=model, horizon='90 days', cutoffs=cutoffs)
df_cv.head()
```
```
from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p
```
ประเมิณความถูกต้องของโมเดลโดยใช้ Library ของ Facebook ที่ชื่อ Performance_metrics โดย Library นี้คำนวนทั้ง MSE,RMSE,MAE,MAPE, และอื่นๆ พร้อมในบรรทัดเดียว โดยผลลัพธ์ที่ได้ คือโมเดลของเรามีค่า MSE อยู่ที่ 4.79 ซึ่งถือได้ว่ามีความแม่นยำ





