# 角矩形的数量
# 给定一个只包含 0 和 1 的网格，找出其中角矩形的数量。
# 一个 角矩形 是由四个不同的在网格上的 1 形成的轴对称的矩形。注意只有角的位置才需要为 1。并且，4 个 1 需要是不同的。

# 思路 遍历上下边,然后找侧边,统计
class Solution:
    def countCornerRectangles(self, grid: List[List[int]]) -> int:
        row = len(grid)
        col = len(grid[0])
        res = 0
        for i1 in range(row):
            for i2 in range(i1+1,row):
                cnt = 0
                for j in range(col):
                    if grid[i1][j] and grid[i2][j]:
                        cnt+=1
                res += ((cnt-1)*cnt//2)
        return res

# 比特位计数
# 给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
# 观察数组特点
# [0 1 1 2 1 2 2 3 1 2 2 3 2 3 3 4]
#  -
#  ---
#  -------
#  ---------------
#  -------------------------------
# 划线部分后一段均为前一段对位的数字加1
# res[cnt+i] = res[cnt]+1
class Solution:
    def countBits(self, num: int) -> List[int]:
        k=1
        while 2**k<num:
            k+=1
        p = 2**k+1
        res = [0 for _ in range(p)]
        cnt = 0
        i = 1
        while i<p-1:
            while cnt<i:
                res[cnt+i] = res[cnt]+1
                cnt+=1
            cnt=0
            i*=2
        res[p-1]=1
        return res[:num+1]


# 最长回文子串
# 动态规划
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s :
            return ""
        res = ""
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        max_len = float("-inf")
        for i in range(n):
            for j in range(i + 1):
                if s[i] == s[j] and (i - j <= 2 or dp[j + 1][i - 1]):
                    dp[j][i] = 1
                if dp[j][i] and  max_len < i + 1 - j:
                    res = s[j : i + 1]
                    max_len = i + 1 - j
        return res


# Manacher算法
# O(n)复杂度
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s)<=1:
            return s
        l = 2*len(s)+1
        s_new = ['#']
        for i in range(0,l-2,2):
            s_new.append(s[i//2])
            s_new.append('#')
        p = [0 for _ in range(l)]
        mx,iid = 0,0
        res = s[0]
        maxx = 1
        for i in range(l):
            if i<mx:
                p[i] = min(p[2*iid-i],mx-i)
            elif i==mx:
                p[i]=1
            while i - p[i] >= 0 and i + p[i] < l and s_new[i - p[i]] == s_new[i + p[i]]:
                p[i]+=1
            if i+p[i]>mx:
                mx = i+p[i]
                iid = i
            if p[i]-1>maxx:
                maxx = p[i]-1
                res = s[(i-p[i]+1)//2:(i+p[i])//2]
        return res

# 最大矩形
# 给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix: return 0

        m = len(matrix)
        n = len(matrix[0])

        left = [0] * n # initialize left as the leftmost boundary possible
        right = [n] * n # initialize right as the rightmost boundary possible
        height = [0] * n

        maxarea = 0

        for i in range(m):

            cur_left, cur_right = 0, n
            # update height
            for j in range(n):
                if matrix[i][j] == '1': height[j] += 1
                else: height[j] = 0
            # update left
            for j in range(n):
                if matrix[i][j] == '1': left[j] = max(left[j], cur_left)
                else:
                    left[j] = 0
                    cur_left = j + 1
            # update right
            for j in range(n-1, -1, -1):
                if matrix[i][j] == '1': right[j] = min(right[j], cur_right)
                else:
                    right[j] = n
                    cur_right = j
            # update the area
            for j in range(n):
                maxarea = max(maxarea, height[j] * (right[j] - left[j]))

        return maxarea

# N个括号的有效组合 动态规划
# 具体思路如下： 当我们清楚所有 i<n 时括号的可能生成排列后，对与 i=n 的情况，我们考虑整个括号排列中最左边的括号。 
# 它一定是一个左括号，那么它可以和它对应的右括号组成一组完整的括号 "( )"，我们认为这一组是相比 n-1 增加进来的括号。
# 那么，剩下 n-1 组括号有可能在哪呢？ 【这里是重点，请着重理解】 剩下的括号要么在这一组新增的括号内部，要么在这一组新增括号的外部（右侧）。
# 既然知道了 i<n 的情况，那我们就可以对所有情况进行遍历： "(" + 【i=p时所有括号的排列组合】 + ")" + 【i=q时所有括号的排列组合】 
# 其中 p + q = n-1，且p q均为非负整数。 事实上，当上述 p 从 0 取到 n-1，q 从 n-1 取到 0 后，所有情况就遍历完了。

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n==0:
            return []
        total_l = []
        total_l.append([None])    # 0组括号时记为None
        total_l.append(["()"])    # 1组括号只有一种情况
        for i in range(2,n+1):    # 开始计算i组括号时的括号组合
            l = []        
            for j in range(i):    # 开始遍历 p q ，其中p+q=n-1 , j 作为索引
                now_list1 = total_l[j]    # p = j 时的括号组合情况
                now_list2 = total_l[i-1-j]    # q = (i-1) - j 时的括号组合情况
                for k1 in now_list1:  
                    for k2 in now_list2:
                        if k1 == None:
                            k1 = ""
                        if k2 == None:
                            k2 = ""
                        el = "(" + k1 + ")" + k2
                        l.append(el)    # 把所有可能的情况添加到 l 中
            total_l.append(l)    # l这个list就是i组括号的所有情况，添加到total_l中，继续求解i=i+1的情况
        return total_l[n]

