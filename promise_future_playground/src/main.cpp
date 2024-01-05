/*
 * A promise is an object that can store a value of type T to be retrieved by a future object
 *   that can possibly be in another thread, offering a synchronization point
 *
 * The promise object is the asynchronous provider and is expected to set a value for the 
 *   shared state at some point
 *
 * The future object is an asynchronous return object that can retrieve the value of the shared
 *   state, waiting for it to be ready, if necessary 
 *
 * Calling future::get() on a valid future blocks the thread until the provider makes the shared
 *   state ready 
 *
 * The lifetime of the shared state lasts at least until the last object with which it is 
 *   associated releases it or is destroyed 
 */

#include <iostream>
#include <functional>
#include <thread>
#include <future>
#include <chrono>
#include <numeric>

void print_int(std::future<int>& fut) {
    int x = fut.get();
    std::cout << "Value: " << x << std::endl;
}

void accumulate(std::vector<int>::iterator first,
                std::vector<int>::iterator last,
                std::promise<int> accumulate_promise) {
    int sum = std::accumulate(first, last, 0);
    accumulate_promise.set_value(sum);
}

void do_work(std::promise<void> barrier) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    barrier.set_value();
}

int main() {
    // Example 1
    std::promise<int> prom;
    std::future<int> fut = prom.get_future(); // Engages with future
                                              //
    std::thread th1 (print_int, std::ref(fut)); // Send this future to the thread
                                                //
    prom.set_value(10);
    th1.join();

    // Example 2
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6};
    std::promise<int> accumulate_promise;
    std::future<int> accumulate_future = accumulate_promise.get_future();
    std::thread work_thread {accumulate, 
                             numbers.begin(), 
                             numbers.end(), 
                             std::move(accumulate_promise)};
    int result = accumulate_future.get();
    std::cout << "Result: " << result << std::endl;
    work_thread.join();

    // Example 3
    std::promise<void> barrier;
    std::future<void> barrier_future = barrier.get_future();
    std::thread new_work_thread {do_work, std::move(barrier)};
    barrier_future.wait();
    new_work_thread.join();
}
