# lineer_regresyon_anlatim
Makine öğrenmesi dersinde işlenen lineer regresyon python kodu ve anlatımı


Lineer regresyon uygulama: 

Öncelikle matematiksel altyapıya bir dalalım 

Lineer bir fonksiyon oluşturmanın genel formülü bildiğimiz gibi: a+bx şeklindedir ve kordinat sisteminde düz (yani lineer) bir doğru oluşturur.  

Şimdi bizim amacımız veri setindeki tüm bağımsız (x) değişkenleri aynı lineer fonksiyona dahil edip aralarındaki korelasyon ile bağımlı (y) değerleri tahmin etmek diye özetleyebiliriz. 

Dolaysyıyla bizim a+bx fonksiyonunu a+b(tüm x değerleri) gibi bir formata çevirmemiz lazım.  

İşte! Bunun için matrislerden yararlanmamız lazım. Yani x değerlerini matris formuna getirip tüm bu x değerlerine karşılık gelebilecek katsayıları yazabilirsek işte o zaman tek bir lineer fonksityon ile aradaki korelasyonu bulabilirz. 

Temel fikrimiz bu 

Şimdi bu katsayılara a,b demek yerine b0, b1 dersek, genel formülümüz:  

ŷ = b0 + b1x...bnxn +e olur (burada “ŷ” tahmin edilen değer,e ise hata değeri) 

Şimdi genel formüllü oturttuk. Peki katsayıları nasıl bulacağız.  

Artık regresyon kısmının “makine öğrenmesi” kısmına geçiş yaptık diyebiliriz. 

Amacımız minimum düzeyda hata ile katsayıları yerleştirmek olduğu için makineye öğrenemsi gereken bir değer vermeliyiz. İşte bu değer de bizim hata fonksiyonumuz yani ∑n i=1,(yi- ŷi)^2 olur. 

Gradyan azalması: 

Bizim bu hata fonksiyonumuzu minimuma düşürmek için yapabileceğimiz en yaygın ve geçerli yöntem, gradyan azalmasıdır. Bunu yapmak için bu hata fonksiyonumuzun türevini bulup bunu gradyanlarımız için kullanmamız lazım. Şimdi gelin bu terimler nedir nasıl bulunur bakalım. 

Öncelikle gradyanları bulmak için hata fonksiyonunun kısmı türevlerini bulmamız lazım. 

∂J / ∂β0 =−2(i=1)∑n (yi −(β0 +β1 xi ))= −2(i=1)∑n (yi − ŷi) 

Bu b0 a göre aldığımız kısmi türev b0’ın hataya etkisi olduğu için b0 katsayısı için bulduğumuz gradyan olacaktır. 

Aynısını b1 ve diğer x e bağlı katsayılar için yaptığımız zaman gradyan: 

−2xi(i=1)∑n (yi − ŷi) olacaktır. 

Bu gradyanları iteratif şekilde tekrar tekrar hesaplayarak gitgide daha doğru katsayılar buluruz ve böylece “makinaye öğretmiş” oluruz. 

Şimdi bunları python üzerinde kod yazarak pekiştirelim. 

import numpy as np import matplotlib.pyplot as plt 

 

X = np.array([20, 30, 40]) y = np.array([3000, 5000, 10000]) 

alpha = 0.00001 # Öğrenme oranı iterations = 200 # Iterasyon sayısı 

beta0, beta1 = 0, 0 

for i in range(iterations): # Tahmin y_pred = beta0 + beta1 * X 

grad_beta0 = -2 * np.sum(y - y_pred) 
grad_beta1 = -2 * np.sum(X * (y - y_pred)) 
 
 
beta0 = beta0 - alpha * grad_beta0 
beta1 = beta1 - alpha * grad_beta1 
 
# Her 10 iterasyonda bir sonucu yazdır 
if i % 10 == 0: 
    print(f"Iterasyon {i}: β0 = {beta0:.2f}, β1 = {beta1:.2f}") 
  

print(f"Final Katsayılar: β0 = {beta0:.2f}, β1 = {beta1:.2f}") 

 

Bu işlemleri aynı zamanda kapalı çözüm ile de yapabilirz bu öğrenme yönteminde daha kesin ve hızlı sonuç verevektir. Ama tabiki her regresyon veya tahmin yönteminin denklemi çözülemeyebilir. Bu daha basit olan lineer regresyon için geçerlidir. 

Bu çözümde de yine hata fonksiyonunu kullancağız ve türevini sıfıra eşitliyerek hatanın “çukur noktasını bulacağız. 

Hata fonk. Türevi: XTXβ=XTy 

 

a) X T X: Bu matris, bağımsız değişkenler (x) arasındaki ilişkiyi özetler. Özellikle, X T X'in elemanları, x değerlerinin karelerinin toplamını ve çarpımlarını içerir.  

b) X T y: Bu matris, bağımlı değişkenler (y) ile bağımsız değişkenler (x) arasındaki ilişkiyi ölçer. Özellikle, X T y'nin elemanları, x ve y değerlerinin çarpımlarının toplamını içerir. c) Denklem: X T Xβ=X T y, veriler arasındaki ilişkiyi kullanarak katsayıları (β) bulmamızı sağlar. Bu denklem, hataların karelerinin toplamını minimize eden β değerlerini verir. 

Şimdi bunu da pythonda kendimiz yazarak pekiştirelim ve sonuçları karşılaştıralım. 

 

import numpy as np import matplotlib.pyplot as plt 

x = [20, 30, 40,80] y = np.array([3000, 5000, 10000,15000])  

ones_column = np.ones((len(x), 1)) x_t = np.array(x).reshape(-1, 1) X_Transformed = np.hstack((ones_column, x_t)) 

X_T_X = X_Transformed.T @ X_Transformed X_T_y = X_Transformed.T @ y 

coefficients = np.linalg.inv(X_T_X) @ X_T_y 

newY = coefficients[0] + coefficients[1] * np.array(x) 

print("katsayılar:", coefficients) print("tahmin Y değerleri :", newY) 

 

 
