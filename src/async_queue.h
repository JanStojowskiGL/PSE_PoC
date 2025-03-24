#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

template <typename T> class AsyncQueue {
public:
  explicit AsyncQueue(size_t capacity) : capacity_(capacity) {}

  // Blocking push: waits until there is room in the queue.
  void push(const T &item) {
    std::unique_lock<std::mutex> lock(mutex_);
    // Wait until queue size is below the capacity
    cv_not_full_.wait(lock, [this]() { return queue_.size() < capacity_; });
    queue_.push(item);
    // Notify potential pop() callers (if waiting)
    cv_not_empty_.notify_one();
  }

  // Overload for rvalue references
  void push(T &&item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_full_.wait(lock, [this]() { return queue_.size() < capacity_; });
    queue_.push(std::move(item));
    cv_not_empty_.notify_one();
  }

  // Non-blocking pop: returns std::optional<T>
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty())
      return std::nullopt; // Nothing to pop, return empty optional

    T item = std::move(queue_.front());
    queue_.pop();
    // Notify potential push() callers that space is available
    cv_not_full_.notify_one();
    return item;
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_not_full_;
  std::condition_variable cv_not_empty_;
  std::queue<T> queue_;
  size_t capacity_;
};
