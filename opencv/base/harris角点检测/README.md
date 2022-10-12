#### harris角点检测

一个图像里面的内容,一般可以分成3种类型:

* 内容没有什么特别变化的平面区域,例如:天空,大海
* 内容在单一方向上变化较大的边缘轮廓区域,例如:房屋边沿
* 内容在多个方向上变化较大的拐角(角点)区域,例如:房屋的边角

harris算法角点检测就可以区分出图像中的这三种区域,特别是角点区域.因为
角点区域可以算是一种很重要的区域特征.harris算法是通过检测区域的自相似性函数来判断这个检测区域是不是角点的:

* 如果检测区域不管往那个方向进行移动,检测区域的灰度值变化都很低的话,
  那么这个区域的自相似函数的值就很低了,就可以知道这是个平面区域了.
* 如果检测区域往多个方向移动,只有一个方向上灰度值有明显变化,那么这个检测区域
  就是边沿区域
* 如果检测区域往多个方向移动,多个方向的灰度值都有明显的变化,那么这个区域就
  是角点区域了

##### 数学原理

对于某一个图像区域$\psi(x,y)$,在它平移了$(\Delta x,\Delta y)$之后,他的自关联函数,我们可以这样表示:

$$
c(x,y,\Delta x,\Delta y) = \sum_{(u,v)\in W(x,y)}w(u,v)(I(u,v)-I(u+\Delta x,v+\Delta y))^2 \tag{$I(u,v)$表示这里的灰度值}
$$

然后我们可以对$I(u+\Delta x,v+\Delta y)$进行泰勒展开,取一阶的近似:

$$
\begin{aligned}
&I(u+\Delta x,v+\Delta y) \\
&=I(u,v)+I_x(u,v)\Delta x+I_y(u,v)\Delta y+o(\Delta x,\Delta y) \\
&\approx I(u,v)+I_x(u,v)\Delta x+I_y(u,v)\Delta y \\
\end{aligned}
\tag{$I_x,I_y$为(x,y)的一阶导}
$$

那么自关联函数就可以化简成:

$$
c(x,y,\Delta x,\Delta y) \approx \sum_{(u,v)\in W(x,y)}w(u,v)(I_x(u,v)\Delta x+I_y(u,v)\Delta y)^2
$$

然后我们把这个式子转化成矩阵的形式:

$$
\begin{aligned}
&c(x,y,\Delta x,\Delta y) \approx [\Delta x,\Delta y]M(x,y)
\left[
\begin{matrix}
\Delta x \\
\Delta y
\end{matrix}
\right] \\
&其中 M(x,y)
=\sum_w
\left[
\begin{matrix}
I_x(x,y)^2 & I_x(x,y)I_y(x,y) \\
I_x(x,y)I_y(x,y) & I_y(x,y)^2
\end{matrix}
\right]
=
\left[
\begin{matrix}
\sum_w I_x(x,y)^2 & \sum_w I_x(x,y)I_y(x,y) \\
\sum_w I_x(x,y)I_y(x,y) & \sum_w I_y(x,y)^2
\end{matrix}
\right]
=
\left[
\begin{matrix}
A & C \\
C & B
\end{matrix}
\right]
\end{aligned}
$$

现在可以看到矩阵M是一个实对称阵,那么它就可以做对角化(看看线代?):

$$
\left[
\begin{matrix}
A & C \\
C & B
\end{matrix}
\right]
=
\left[
\begin{matrix}
\lambda_1 & 0 \\
0 & \lambda_2
\end{matrix}
\right]
$$

然后自关联函数就可以转换成:

$$
c(x,y,\Delta x,\Delta y) \approx A\Delta x^2+2C\Delta x \Delta y+B\Delta y^2 = \lambda_1\Delta x^2+\lambda_2\Delta y^2
$$

所以最后自关联函数就是一个椭圆函数了,椭圆的两个轴$\lambda_1,\lambda_2$决定了这个椭圆的大小,即$c$的大小,而$\lambda_1,\lambda_2$ 通过$A,B,C$ 是可以求得的.

最后就可以下结论了:

* 如果$\lambda_1,\lambda_2$ 都小,并且近似相等的话,这时,自相关函数在各个方向上都小,此时的检测区域是平面
* 如果$\lambda_1,\lambda_2$一个远大于另一个的话,自相关函数就在某一个方向上大,此时的检测区域是边界
* 如果$\lambda_1,\lambda_2$都大,并且近似相等的话,自相关函数就在所有方向都是大的,此时额检测区域就是角点了

##### 角点响应R值

单纯的用$c$的数值去判断是否是角点不是那么好,所以科学家弄了个R值来表示:

$$
R = detM-\alpha(traceM)^2 \tag{其中,$detM$为$\lambda_1\lambda_2$,$traceM$为$\lambda_1+\lambda_2$}
$$

那么就有:

* R为小数值,区域为平面
* R为大数值正数,为角点
* R为大数值负数,为边界
