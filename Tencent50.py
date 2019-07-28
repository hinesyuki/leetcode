# LeetCode一部分题目的整理

# 螺旋矩阵
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        l, r, t, b = 0, n - 1, 0, n - 1
        mat = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n
        while num <= tar:
            for i in range(l, r + 1): # left to right
                mat[t][i] = num
                num += 1
            t += 1
            for i in range(t, b + 1): # top to bottom
                mat[i][r] = num
                num += 1
            r -= 1
            for i in range(r, l - 1, -1): # right to left
                mat[b][i] = num
                num += 1
            b -= 1
            for i in range(b, t - 1, -1): # bottom to top
                mat[i][l] = num
                num += 1
            l += 1
        return mat

# 全排列
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return list(itertools.permutations(nums))

# 子集
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + num for num in res]
        return res

# 二叉树最大深度(拓展 统计二叉树节点)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None: 
            return 0 
        else: 
            left_height = self.maxDepth(root.left) 
            right_height = self.maxDepth(root.right) 
            return max(left_height, right_height) + 1 

# gray编码
class Solution:
    def grayCode(self, n: int) -> List[int]:
        return [i ^ i >> 1  for i in range(1 << n)]

        # dp = [0 for _ in range(2**n)]
        # for i in range(1, 2**n):
        #     dp[i] = i ^ int(i/2)
        # return(dp)


# 二叉搜索树的最近公共祖先
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > q.val:
            p,q = q,p
        while root:
            if root.val > q.val:
                root = root.left
            elif root.val < p.val:
                root = root.right
            else:
                return root

# 二叉搜索树第K小的数字
# 中序遍历 再补充前序和后序
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        res = []
        def middle_digui(root:TreeNode):
            if root==None:
                return
            else:
                middle_digui(root.left)
                res.append(root.val)
                middle_digui(root.right)
        middle_digui(root)
        return res[k-1]



# 求众数 (大于len//2的)
# 解法1：摩尔投票算法
# 这个方法如果要详细的去理解，很麻烦，也分很多种情况。
# 其核心思想就是：抵消
# 最差的情况就是其他所有的数都跟众数做了抵消，但是由于众数出现的次数大于1/2，所以最终剩下的还是众数。

# 解法2:排序中间的
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count=1
        target = nums[0]
        n = len(nums)
        for i in range(1,n):
            if nums[i] == target:
                count += 1
            else:
                if count>=1:
                    count -= 1
                else:
                    target = nums[i]
                    
        return target


# 求回文数 list翻转然后比较
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x<0:
            return False
        x_list = list(str(x))
        y_list = x_list.copy()
        x_list.reverse()
        return(x_list==y_list)

# 二叉树最近公共祖先
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if(root is None or root==p or root==q):
            return root
        left=self.lowestCommonAncestor(root.left,p,q)
        right=self.lowestCommonAncestor(root.right,p,q)
        if left  and right :
            return root
        if not left :
            return right
        if not right :
            return left
        return None

class Solution:

    def __init__(self):
        # Variable to store LCA node.
        self.ans = None

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def recurse_tree(current_node):

            # If reached the end of a branch, return False.
            if not current_node:
                return False

            # Left Recursion
            left = recurse_tree(current_node.left)

            # Right Recursion
            right = recurse_tree(current_node.right)

            # If the current node is one of p or q
            mid = current_node == p or current_node == q

            # If any two of the three flags left, right or mid become True.
            if mid + left + right >= 2:
                self.ans = current_node

            # Return True if either of the three bool values is True.
            return mid or left or right

        # Traverse the tree
        recurse_tree(root)
        return self.ans

# 多少路径
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        a1 = [1 for _ in range(n)]
        a2 = [1 for _ in range(n)]
        for i in range(1, m):
            for j in range(1, n):
                a2[j] = a1[j] + a2[j-1]
            a1 = a2.copy()
        return a1[-1]


# 数组是否存在重复元素
# 利用python set的不含重复元素性
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len((set(nums))) != len(nums)


# 实现最小栈
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack_list = []
        self.min_list = []
        self.min = None

    def push(self, x: int) -> None:
        if not self.stack_list:
            self.min = x
            self.min_list.append(self.min)
            self.stack_list.append(x)
            return
        self.stack_list.append(x)
        if self.min >= x:
            self.min = x
            self.min_list.append(self.min)
        return
        
    def pop(self) -> None:
        pop_result = None
        if self.stack_list:
            pop_result = self.stack_list[-1]
            if self.stack_list.pop() == self.min:
                self.min_list.pop()
                if self.min_list:
                    self.min = self.min_list[-1]
                else:
                    self.min = None
            return pop_result
        else:
            self.min = None
            return pop_result
        
    def top(self) -> int:
        a = self.stack_list.pop()
        self.stack_list.append(a)
        return a

    def getMin(self) -> int:
        return self.min
# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()



# 合并K个有序链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if len(lists)==0:
            return None
        amount = len(lists)
        interval = 1
        while interval < amount:
            for i in range(0, amount - interval, interval * 2):
                lists[i] = self.mergeTwoLists(lists[i], lists[i + interval])
            interval *= 2
        return lists[0] if amount > 0 else lists
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        res_list = ListNode(0)
        ans_list = res_list
        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                res_list.next = l1
                res_list = res_list.next
                l1 = l1.next
            else:
                res_list.next = l2
                res_list = res_list.next
                l2 = l2.next
        if l1 is not None:
            res_list.next = l1
        if l2 is not None:
            res_list.next = l2
        return ans_list.next

# 最大子序和
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums)== 1:
            return nums[0]
        res = nums[0]
        s = 0
        for i in range(len(nums)):
            s += nums[i]
            if s>res:
                res = s
            if s<0:
                s = 0
        return res


# 相交链表
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        a = []
        b = []
        while headA is not None:
            a.append(headA)
            headA = headA.next
        while headB is not None:
            b.append(headB)
            headB = headB.next
        a.reverse()
        b.reverse()
        l = min(len(a),len(b))
        res = None
        for i in range(l):
            if a[i].val == b[i].val and a[i]==b[i]:
                res = a[i]
            else:
                return res
        return res


# LRU缓存机制
# dict记录key list记录use_time先后
class LRUCache:

    def __init__(self, capacity: int):
        self.lru_dict = {}
        self.use_list = []
        self.capacity = capacity
        
    def get(self, key: int) -> int:
        if key in self.lru_dict:
            if key in self.use_list:
                self.use_list.remove(key)
            self.use_list.append(key)
            return self.lru_dict[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.lru_dict or len(self.lru_dict)<self.capacity:
            self.lru_dict[key] = value
            if key in self.use_list:
                self.use_list.remove(key)
            self.use_list.append(key)
        elif len(self.lru_dict)==self.capacity:
            del self.lru_dict[self.use_list[0]]
            self.use_list.remove(self.use_list[0])
            self.use_list.append(key)
            self.lru_dict[key] = value
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


# orderdict
from collections import OrderedDict
# 创建OrderedDict对象
dx = OrderedDict(b=5, c=2, a=7)
print(dx) # OrderedDict([('b', 5), ('c', 2), ('a', 7)])
d = OrderedDict()
# 向OrderedDict中添加key-value对
d['Python'] = 89
d['Swift'] = 92
d['Kotlin'] = 97
d['Go'] = 87
# 遍历OrderedDict的key-value对
for k,v in d.items():
    print(k, v)

# 创建普通的dict对象
my_data = {'Python': 20, 'Swift':32, 'Kotlin': 43, 'Go': 25}
# 创建基于key排序的OrderedDict
d1 = OrderedDict(sorted(my_data.items(), key=lambda t: t[0]))
# 创建基于value排序的OrderedDict
d2 = OrderedDict(sorted(my_data.items(), key=lambda t: t[1]))
print(d1) # OrderedDict([('Go', 25), ('Kotlin', 43), ('Python', 20), ('Swift', 32)])
print(d2) # OrderedDict([('Python', 20), ('Go', 25), ('Swift', 32), ('Kotlin', 43)])
print(d1 == d2) # False

d = OrderedDict.fromkeys('abcde')
# 将b对应的key-value对移动到最右边（最后加入）
d.move_to_end('b')
print(d.keys()) # odict_keys(['a', 'c', 'd', 'e', 'b'])
# 将b对应的key-value对移动到最左边（最先加入）
d.move_to_end('b', last=False)
print(d.keys()) # odict_keys(['b', 'a', 'c', 'd', 'e'])
# 弹出并返回最右边（最后加入）的key-value对
print(d.popitem()[0]) # e
# 弹出并返回最左边（最先加入）的key-value对
print(d.popitem(last=False)[0]) # b


# 环形链表 破坏结构和不破坏结构两种
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 破坏结构的 置空法
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        while head.next and head.val is not None:
            head.val = None
            head = head.next
        return True if head.next else False

# 非破坏结构的
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node_list = set()
        while head:
            if head in node_list:
                return head
            else:
                node_list.add(head)
                head = head.next
        return head

# 最接近目标的三数之和
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # 我的慢的
        nums.sort()
        minn = float("inf")
        res = 0
        for k in range(len(nums)):
            a = nums.copy()
            a.remove(nums[k])
            i,j = 0,len(nums)-2
            p = target - nums[k]
            while i<j:
                temp = p-a[i]-a[j]
                if abs(minn) > abs(temp):
                    minn = temp
                    res = a[i]+a[j]+nums[k]
                if temp>0:
                    i+=1
                else:
                    j-=1
        return res
        # 别人的快的
        # n=len(nums)
        # nums.sort()
        # res=float('inf')
        # for raw in range(n-2):
        #     if raw>0 and nums[raw]==nums[raw-1]:continue
        #     i,j=raw+1,n-1
        #     while i<j:
        #         a=nums[raw]+nums[i]+nums[j]
        #         if a>target:
        #             if abs(a-target)<abs(res):
        #                 res=(target-a)
        #             j-=1
        #         elif a<target:
        #             if abs(a-target)<abs(res):
        #                 res=(target-a)
        #             i+=1
        #         else: return target
        # return target-res


# 最大路径和可能有三种情况 :
# 1.在左子树内部
# 2.在右子树内部
# 3.在穿过左子树，根节点，右子树的一条路径中

# 设计一个递归函数，返回以该节点为根节点，单向向下走的最大路径和
# 注意的是，如果这条路径和小于0的话，则置为0，就相当于不要这条路了，要了也是累赘

# int left = helper(root.left)
# int right = helper(root.right)

# 我们递归调用这个函数，则最大值为 left + right + root.val，我们用一个全局变量不断更新它
# 然后这个递归函数返回的就是 max(left, right) + root.val，也就是以这个节点为根节点向下走的最大路径和
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def max_gain(node):
            nonlocal max_sum
            if not node:
                return 0

            # max sum on the left and right sub-trees of node
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            # the price to start a new path where `node` is a highest node
            price_newpath = node.val + left_gain + right_gain
            
            # update max_sum if it's better to start a new path
            max_sum = max(max_sum, price_newpath)
        
            # for recursion :
            # return the max gain if continue the same path
            return node.val + max(left_gain, right_gain)
   
        max_sum = float('-inf')
        max_gain(root)
        return max_sum

# 旋转链表差值
# 先找到旋转中心,再查值
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def find_min(left, right):
            if nums[left] < nums[right]:
                return 0
            
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] > nums[mid + 1]:
                    return mid + 1
                else:
                    if nums[mid] < nums[left]:
                        right = mid - 1
                    else:
                        left = mid + 1
                
        def search(left, right):
            """
            Binary search
            """
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] == target:
                    return mid
                else:
                    if target < nums[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1
            return -1
        
        n = len(nums)
        
        if n == 0:
            return -1
        if n == 1:
            return 0 if nums[0] == target else -1 
        
        rotate_index = find_min(0, n - 1)
        
        # if target is the smallest element
        if nums[rotate_index] == target:
            return rotate_index
        # if array is not rotated, search in the entire array
        if rotate_index == 0:
            return search(0, n - 1)
        if target < nums[0]:
            # search on the right side
            return search(rotate_index, n - 1)
        # search on the left side
        return search(0, rotate_index)
            
# 两个有序链表的中位数
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        if n1 > n2:
            return self.findMedianSortedArrays(nums2,nums1)
        k = (n1 + n2 + 1)//2
        left = 0
        right = n1
        while left < right :
            m1 = left +(right - left)//2
            m2 = k - m1
            if nums1[m1] < nums2[m2-1]:
                left = m1 + 1
            else:
                right = m1
        m1 = left
        m2 = k - m1 
        c1 = max(nums1[m1-1] if m1 > 0 else float("-inf"), nums2[m2-1] if m2 > 0 else float("-inf") )
        if (n1 + n2) % 2 == 1:
            return c1
        c2 = min(nums1[m1] if m1 < n1 else float("inf"), nums2[m2] if m2 <n2 else float("inf"))
        return (c1 + c2) / 2