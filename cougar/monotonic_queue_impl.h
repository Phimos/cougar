#ifdef T

#include "math.h"
#include "stdlib.h"

#define concat(a, b) a##_##b

#define monotonic_queue(dtype) struct concat(monotonic_queue, dtype)
#define method(name, dtype) concat(monotonic_queue_##name, dtype)

monotonic_queue(T) {
    size_t size, capacity;
    size_t front, back;
    size_t increasing;
    T* buffer;
    size_t* indices;
};

static inline void method(reset, T)(monotonic_queue(T) * queue) {
    queue->size = 0;
    queue->front = 0;
    queue->back = 0;
}

static inline monotonic_queue(T) * method(init, T)(size_t capacity, size_t increasing) {
    monotonic_queue(T)* queue = (monotonic_queue(T)*)malloc(sizeof(monotonic_queue(T)));
    queue->capacity = capacity;
    queue->increasing = increasing;
    queue->buffer = (T*)malloc(sizeof(T) * capacity);
    queue->indices = (size_t*)malloc(sizeof(size_t) * capacity);
    method(reset, T)(queue);
    return queue;
}

static inline void method(free, T)(monotonic_queue(T) * queue) {
    free(queue->buffer);
    free(queue->indices);
    free(queue);
}

static inline size_t method(next, T)(monotonic_queue(T) * queue, size_t index) {
    return ((index + 1) < queue->capacity) ? (index + 1) : 0;
}

static inline size_t method(prev, T)(monotonic_queue(T) * queue, size_t index) {
    return (index > 0) ? (index - 1) : (queue->capacity - 1);
}

static inline T method(front_value, T)(monotonic_queue(T) * queue) {
    return queue->buffer[queue->front];
}

static inline size_t method(front_index, T)(monotonic_queue(T) * queue) {
    return queue->indices[queue->front];
}

static inline T method(back_value, T)(monotonic_queue(T) * queue) {
    return queue->buffer[queue->back];
}

static inline size_t method(back_index, T)(monotonic_queue(T) * queue) {
    return queue->indices[queue->back];
}

static inline void method(pop_front, T)(monotonic_queue(T) * queue) {
    queue->front = method(next, T)(queue, queue->front);
    queue->size--;
}

static inline void method(pop_back, T)(monotonic_queue(T) * queue) {
    queue->back = method(prev, T)(queue, queue->back);
    queue->size--;
}

static inline void method(push_back, T)(monotonic_queue(T) * queue, T value, size_t index) {
    if (queue->size == 0) {
        queue->front = queue->back = 0;
    } else {
        queue->back = method(next, T)(queue, queue->back);
    }
    queue->buffer[queue->back] = value;
    queue->indices[queue->back] = index;
    queue->size++;
}

static inline void method(push, T)(monotonic_queue(T) * queue, T value, size_t index) {
    if (queue->increasing) {
        if (queue->size > 0 && method(front_value, T)(queue) >= value) {
            method(reset, T)(queue);
        } else {
            while (queue->size > 0 && method(back_value, T)(queue) >= value) {
                method(pop_back, T)(queue);
            }
        }
    } else {
        if (queue->size > 0 && method(front_value, T)(queue) <= value) {
            method(reset, T)(queue);
        } else {
            while (queue->size > 0 && method(back_value, T)(queue) <= value) {
                method(pop_back, T)(queue);
            }
        }
    }
    method(push_back, T)(queue, value, index);
}

#undef method
#undef monotonic_queue
#undef concat

#endif