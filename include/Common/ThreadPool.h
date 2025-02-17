#include <future>
#include <vector>
#include <functional>
#include <mutex>
#include "Common/Utils.h"
#include <queue>

namespace KernelCodeGen {

class Worker;
class ThreadPool {
public:
    friend class Worker;
    explicit ThreadPool(int workerCount = 5);
    void init();
    void push_task(std::function<KernelInfo(Config)> task, Config cfg);
    void wait_finish(int taskCount);
    void stop_workers();
    std::vector<KernelInfo> get_result();
private:
    void incrWorkerCount();
    void decrWorkerCount();
    void pushResult(KernelInfo result);
    int m_validWorkerCount;
    int m_maxWorkerCount;
    std::mutex m_taskQueueLock;
    std::condition_variable m_cvSchedule;  // m_validWorkerCount > 0
    std::queue<std::future<KernelCodeGen::KernelInfo>> m_tasks;
    std::vector<Worker*> m_workers;
    std::vector<KernelInfo> m_results;
    std::mutex m_resultsLock;  // m_results
    std::mutex m_scheduleLock;  // valid worker count
    std::vector<std::future<void>> m_workerFutures;
    bool m_startFlag = false;
};

class Worker {
public:
    explicit Worker(ThreadPool* pool);
    void run();
    void stop();
private:
    ThreadPool* m_pool;

};

}
