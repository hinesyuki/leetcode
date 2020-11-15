#define MAX_SIZE 512

struct User {
    int id;
    int followee[MAX_SIZE];
    struct User* next;
};

struct Tweet {
    int userId;
    int tweetId;
    struct Tweet* next;
};

typedef struct {
    struct User* user;
    struct Tweet* tweet;
} Twitter;

/** Initialize your data structure here. */

Twitter* twitterCreate() {
    Twitter* twitter = (Twitter*)malloc(sizeof(Twitter));
    twitter->user = (struct User*)malloc(sizeof(struct User));
    twitter->user->next = NULL;
    twitter->tweet = (struct Tweet*)malloc(sizeof(struct Tweet));
    twitter->tweet->next = NULL;
    return twitter;
}

/** Compose a new tweet. */
void twitterPostTweet(Twitter* obj, int userId, int tweetId) {
    struct Tweet* tweet = (struct Tweet*)malloc(sizeof(struct Tweet));
    tweet->userId = userId;
    tweet->tweetId = tweetId;
    tweet->next = obj->tweet->next;
    obj->tweet->next = tweet;
}

/** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
int* twitterGetNewsFeed(Twitter* obj, int userId, int* retSize) {
  int* ret = (int*)calloc(10, sizeof(int));
  *retSize = 0;
  struct Tweet* tweet = obj->tweet->next;
  struct User* user = obj->user->next;
  while (user && user->id != userId) user = user->next;  
  if (user == NULL) {                                   
    while (tweet && *retSize < 10) {
      if (tweet->userId == userId) 
        ret[(*retSize)++] = tweet->tweetId;
      tweet = tweet->next;
    }
    return ret;
  }
  while (tweet && *retSize < 10) {
    if (tweet->userId == userId || user->followee[tweet->userId] == 1) 
      ret[(*retSize)++] = tweet->tweetId;
    tweet = tweet->next;
  }
  return ret;
}

/** Follower follows a followee. If the operation is invalid, it should be a no-op. */
void twitterFollow(Twitter* obj, int followerId, int followeeId) {
    struct User* u = obj->user;
    while (u->next && u->id != followerId) u = u->next;
    if (u->id == followerId)
        u->followee[followeeId] = 1;
    else {
        struct User* user = (struct User*)malloc(sizeof(struct User));
        user->id = followerId;
        user->followee[followeeId] = 1;
        user->next = obj->user->next;
        obj->user->next = user;
    }
}

/** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
void twitterUnfollow(Twitter* obj, int followerId, int followeeId) {
    struct User* u = obj->user;
    while (u->next && u->id != followerId) 
        u = u->next;
    if (u->id == followerId)
        u->followee[followeeId] = 0;
}

void twitterFree(Twitter* obj) {
    if (obj && obj->user) free(obj->user);
    if (obj && obj->tweet) free(obj->tweet);
    if (obj) free(obj);
}

/**
 * Your Twitter struct will be instantiated and called as such:
 * Twitter* obj = twitterCreate();
 * twitterPostTweet(obj, userId, tweetId);
 
 * int* param_2 = twitterGetNewsFeed(obj, userId, retSize);
 
 * twitterFollow(obj, followerId, followeeId);
 
 * twitterUnfollow(obj, followerId, followeeId);
 
 * twitterFree(obj);
*/


#define MaxSize 40

struct SetType{
    char* name;
    int size;   //名字的大小
};
typedef struct SetType SetType;

struct my_struct{
    char* name;
    int pos;    //此名在集合数组中的下标
    UT_hash_handle hh;
};

int Find(int* S, int x){    //路径压缩
    if(S[x] < 0)
        return x;
    else
        return S[x] = Find(S, S[x]);
}

void Union(int* S, SetType* T, int root1, int root2){   //按字典序合并，前提是不同父亲
    if(strcmp(T[root1].name, T[root2].name) < 0){   //判断字典序
        S[root1] += S[root2];
        S[root2] = root1;
        }
    else{
        S[root2] += S[root1];
        S[root1] = root2;
        }
}

char** trulyMostPopular(char** names, int namesSize, char** synonyms, int synonymsSize, int* returnSize){
    struct my_struct *s = NULL, *users = NULL;
    *returnSize = 0;
    char** res = (char**)calloc(namesSize, sizeof(char*));
    SetType* T = (SetType*)calloc(namesSize, sizeof(SetType));
    int* S = (int*)calloc(namesSize, sizeof(int));
    int empty = 0;  //空闲位置
    for(int i = 0; i < namesSize; i++){
        char* tmp = names[i];
        int count = 0;  //出现频率
        int l = 0;
        while(tmp[l] != '(')
            l++;
        int r = l + 1;
        while(tmp[r] != ')'){
            count = count * 10 + tmp[r] - '0';
            r++;
            }
        tmp[l] = '\0';
        s = (struct my_struct *)malloc(sizeof(struct my_struct));
        s->name = names[i];
        s->pos = empty;
        HASH_ADD_STR(users, name, s);
        T[empty].name = names[i];
        T[empty].size = l;
        S[empty] = -count;
        empty++;
        }
    
    struct my_struct *p = NULL;
    for(int i = 0; i < synonymsSize; i++){
        char* tmp = synonyms[i];
        char *left, *right; //分割两个名字
        int l = 1;
        while(tmp[l] != ',') l++;
        int r = l;
        while(tmp[r] != ')') r++;
        left = &(tmp[1]);
        right = &(tmp[l + 1]);
        tmp[l] = tmp[r] = '\0';
        HASH_FIND_STR(users, left, p);
        if(!p) continue;
        int root1 = Find(S, p->pos);
        HASH_FIND_STR(users, right, p);
        if(!p) continue;
        int root2 = Find(S, p->pos);
        if(root1 == root2) continue;    //这里的判断特别重要，如果已经是同一个父亲了就不要合并了
        Union(S, T, root1, root2);
        }

    for(int i = 0; i < empty; i++){
        if(S[i] < 0){
            res[(*returnSize)] = (char*)calloc(MaxSize, sizeof(char));
            sprintf(res[(*returnSize)], "%s(%d)", T[i].name, -S[i]);
            (*returnSize)++;
            }
        }
    free(S);
    free(T);
    //HASH_CLEAR(hh, users);    //不用全局变量的话其实不清除也可以，不然很花时间
    return res;
}