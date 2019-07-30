# 无重复字符的最长子串 

# 滑动窗口
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s == '':
            return 0
        window = set()
        left = 0 
        max_len = 0
        cur_len = 0
        
        for ch in s:
            cur_len += 1
            while ch in window:# 从前向后删除，直到删除了ch
                window.remove(s[left])
                left += 1
                cur_len -= 1
            if cur_len > max_len:
                max_len = cur_len
            window.add(ch)
        return max_len

# 差不多滑动窗口
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        s = "aabcdabcde"
        s_list = list(s)
        # print(s_list)
        res = 0
        res_list = []
        res_dict = {}

        l = 0
        for i in range(len(s_list)):
            if s_list[i] not in res_dict:
                res_list.append(s_list[i])
                res_dict[s_list[i]] = len(res_list)-1
                l += 1
                # print(res_list)
            else:
                res=max(l,res)
                p = res_dict[s_list[i]]+1
                if p==i:
                    res_list = [s_list[i]]
                else:
                    res_list = res_list[p:]
                    res_list.append(s_list[i])
                # print(res_list)
                l-=res_dict[s_list[i]]
                print(i)
                print(res_list)
                print(res_dict)
                res_dict = {value:key for key, value in dict(enumerate(res_list)).items()}
            print('i->%d  l->%d'%(i,l))
            if i==4:
                print(res_dict)
        print(max(l,res))