#include "Common/ThreadPool.h"

namespace KernelCodeGen {

Worker::Worker(ThreadPool* pool):m_pool(pool){
    assert(pool != nullptr);
}

void Worker::stop(){
    ;
}

void Worker::run(){
    auto& taskLock = this->m_pool->m_taskQueueLock;
    while (this->m_pool->m_startFlag)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::future<KernelInfo> taskFuture;
        bool hasTask = false;
        {
            auto lk = std::lock_guard<std::mutex>(taskLock);
            if(!this->m_pool->m_tasks.empty()){
                taskFuture = std::move(m_pool->m_tasks.front());
                m_pool->m_tasks.pop();
                hasTask = true;
            }
        }
        if(hasTask){
            m_pool->decrWorkerCount();
            std::cout << "[worker] start task\n";
            auto retInfo = taskFuture.get();
            this->m_pool->pushResult(retInfo);
            m_pool->incrWorkerCount();
        }
    }

}

void ThreadPool::incrWorkerCount(){
    std::lock_guard<std::mutex> lk(m_scheduleLock);
    this->m_validWorkerCount++;
    m_cvSchedule.notify_all();
}
void ThreadPool::decrWorkerCount(){
    std::lock_guard<std::mutex> lk(m_scheduleLock);
    this->m_validWorkerCount--;
}
void ThreadPool::pushResult(KernelInfo result){
    std::lock_guard<std::mutex> lk(m_resultsLock);
    this->m_results.push_back(result);
}

ThreadPool::ThreadPool(int count) : m_maxWorkerCount(count){
    m_validWorkerCount = m_maxWorkerCount;
    for(int i=0;i < m_maxWorkerCount;++i){
        m_workers.push_back(new Worker(this));
    }
}

void ThreadPool::init(){
    if(this->m_workers.empty()){
        assert(false);
    }
    this->m_startFlag = true;
    for(auto w : m_workers){
        m_workerFutures.push_back(std::async([&](){
            w->run();
        }));
    }
    std::cout << "[pool] init ok\n";
}

void ThreadPool::push_task(std::function<KernelInfo(Config)> task, Config cfg){
    std::unique_lock<std::mutex> lk(m_scheduleLock);
    m_cvSchedule.wait(lk,[&](){
        return this->m_validWorkerCount > 0;
    });
    auto t = std::async(task,cfg);
    {
        std::unique_lock<std::mutex> lk(m_taskQueueLock);
        m_tasks.push(std::move(t));
    }
}

void ThreadPool::stop_workers(){
    this->m_startFlag = false;
    for(auto& w : m_workerFutures){
        w.get();
    }
    m_workerFutures.clear();
}

void ThreadPool::wait_finish(int taskCount){
    while(true){
        std::lock_guard<std::mutex> lk(this->m_resultsLock);
        if(this->m_results.size() == taskCount){
            stop_workers();
            break;
        }
    }
}

std::vector<KernelInfo> ThreadPool::get_result(){
    return this->m_results;
}

}  // end namespace