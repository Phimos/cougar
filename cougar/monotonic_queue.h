#ifndef __MONOTONIC_QUEUE_H__
#define __MONOTONIC_QUEUE_H__

#include "math.h"
#include "stdlib.h"

#include "debug.h"

#ifdef T
#undef T
#endif
#define T double
#include "monotonic_queue_impl.h"
#undef T

#ifdef T
#undef T
#endif
#define T float
#include "monotonic_queue_impl.h"
#undef T

#ifdef T
#undef T
#endif
#define T long long
#include "monotonic_queue_impl.h"
#undef T

#ifdef T
#undef T
#endif
#define T int
#include "monotonic_queue_impl.h"
#undef T

#ifdef T
#undef T
#endif
#define T char
#include "monotonic_queue_impl.h"
#undef T

#define monotonic_queue_ monotonic_queue_double

// struct monotonic_queue_ {
//     size_t size_, capacity_;
//     size_t front_, back_;
//     size_t increasing_;  // 1 for increasing (min), 0 for decreasing (max)

//     double* buffer_;
//     size_t* indices_;
// };

// static inline void monotonic_queue_reset(struct monotonic_queue_* queue) {
//     queue->size_ = 0;
//     queue->front_ = 0;
//     queue->back_ = 0;
// }

// static inline struct monotonic_queue_* monotonic_queue_init(size_t capacity, size_t increasing) {
//     struct monotonic_queue_* queue = (struct monotonic_queue_*)malloc(sizeof(struct monotonic_queue_));
//     queue->buffer_ = (double*)malloc(sizeof(double) * capacity);
//     queue->indices_ = (size_t*)malloc(sizeof(size_t) * capacity);
//     queue->capacity_ = capacity;
//     queue->increasing_ = increasing;
//     monotonic_queue_reset(queue);
//     return queue;
// }

// static inline void monotonic_queue_free(struct monotonic_queue_* queue) {
//     free(queue->buffer_);
//     free(queue->indices_);
//     free(queue);
// }

// static inline size_t monotonic_queue_next(struct monotonic_queue_* queue, size_t index) {
//     return ((index + 1) < (queue->capacity_)) ? (index + 1) : 0;
// }

// static inline size_t monotonic_queue_prev(struct monotonic_queue_* queue, size_t index) {
//     return (index > 0) ? (index - 1) : (queue->capacity_ - 1);
// }

// static inline double monotonic_queue_front_value(struct monotonic_queue_* queue) {
//     return queue->buffer_[queue->front_];
// }

// static inline size_t monotonic_queue_front_index(struct monotonic_queue_* queue) {
//     return queue->indices_[queue->front_];
// }

// static inline double monotonic_queue_back_value(struct monotonic_queue_* queue) {
//     return queue->buffer_[monotonic_queue_prev(queue, queue->back_)];
// }

// static inline size_t monotonic_queue_back_index(struct monotonic_queue_* queue) {
//     return queue->indices_[monotonic_queue_prev(queue, queue->back_)];
// }

// static inline void monotonic_queue_show(struct monotonic_queue_* queue) {
//     debug("[monotonic_queue] show:\n");
//     debug("front: %zu, back: %zu, size: %zu, capacity: %zu, increasing: %zu\n",
//           queue->front_, queue->back_, queue->size_, queue->capacity_, queue->increasing_);
//     for (size_t i = queue->front_; i != queue->back_; i = monotonic_queue_next(queue, i)) {
//         debug("%lf, %zu; ", queue->buffer_[i], queue->indices_[i]);
//     }
//     debug("\n");
// }

// static inline void monotonic_queue_pop_front(struct monotonic_queue_* queue) {
//     queue->front_ = monotonic_queue_next(queue, queue->front_);
//     queue->size_--;
// }

// static inline void monotonic_queue_pop_back(struct monotonic_queue_* queue) {
//     queue->back_ = monotonic_queue_prev(queue, queue->back_);
//     queue->size_--;
// }

// static inline void monotonic_queue_push_back(struct monotonic_queue_* queue, double value, size_t index) {
//     queue->buffer_[queue->back_] = value;
//     queue->indices_[queue->back_] = index;
//     queue->back_ = monotonic_queue_next(queue, queue->back_);
//     queue->size_++;
// }

// static inline void monotonic_queue_push(struct monotonic_queue_* queue, double value, size_t index) {
//     if (queue->increasing_) {
//         if (queue->size_ > 0 && monotonic_queue_front_value(queue) >= value) {
//             monotonic_queue_reset(queue);
//         } else {
//             while ((queue->size_ > 0) && (monotonic_queue_back_value(queue) >= value)) {
//                 monotonic_queue_pop_back(queue);
//             }
//         }
//     } else {
//         if (queue->size_ > 0 && monotonic_queue_front_value(queue) <= value) {
//             monotonic_queue_reset(queue);
//         } else {
//             while ((queue->size_ > 0) && (monotonic_queue_back_value(queue) <= value)) {
//                 monotonic_queue_pop_back(queue);
//             }
//         }
//     }
//     monotonic_queue_push_back(queue, value, index);
// }

#define monotonic_queue_reset monotonic_queue_reset_double
#define monotonic_queue_init monotonic_queue_init_double
#define monotonic_queue_free monotonic_queue_free_double
#define monotonic_queue_next monotonic_queue_next_double
#define monotonic_queue_prev monotonic_queue_prev_double
#define monotonic_queue_front_value monotonic_queue_front_value_double
#define monotonic_queue_front_index monotonic_queue_front_index_double
#define monotonic_queue_back_value monotonic_queue_back_value_double
#define monotonic_queue_back_index monotonic_queue_back_index_double
#define monotonic_queue_pop_front monotonic_queue_pop_front_double
#define monotonic_queue_pop_back monotonic_queue_pop_back_double
#define monotonic_queue_push_back monotonic_queue_push_back_double
#define monotonic_queue_push monotonic_queue_push_double

#endif