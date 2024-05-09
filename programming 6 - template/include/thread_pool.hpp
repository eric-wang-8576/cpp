#include <thread>
#include <queue>
#include <functional>
#include <mutex>

class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    volatile bool stop;

public:
    ThreadPool(int numThreads) : stop(false) {
        // Initialize the pool of threads all attempting to grab tasks from the queue
        for (int i = 0; i < numThreads; ++i) {
            workers.emplace_back(
                // Capture this as a bind
                [this] {
                    while (true) {
                        // Try to grab a task off of the queue
                        std::function<void()> task;

                        // Scope here to take advantage of RAII
                        { 
                            std::unique_lock<std::mutex> lock(this->queue_mutex);

                            // Wait until we stop or there are tasks on the queue
                            this->condition.wait(lock,
                                [this] { return this->stop || !this->tasks.empty(); });

                            if (this->stop && this->tasks.empty()) {
                                return;
                            }

                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }

                        // Perform the task
                        task();
                    }
                }
            );
        }
    }

    ~ThreadPool() {
        // Scope here to take advantage of RAII
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    // Universal reference
    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args) {
        auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        // Scope here to take advantage of RAII
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(task);
        }
        // Let threads know that we have placed a task 
        condition.notify_one();
    }
};
