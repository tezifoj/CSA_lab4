(define xs (makestring 32))
(define count 0)
(define last 0)

(defun digitp (c)
  (if (< c #\0) 0 (if (> c #\9) 0 1)))

(defun skip-spaces (c)
  (if (= c 32) (skip-spaces (readchar)) c))

(defun read-digits (c acc)
  (if (digitp c)
      (read-digits (readchar) (+ (* acc 10) (- c #\0)))
      (progn (setq last c) acc)))

(defun read-list (c)
  (let ((d (skip-spaces c)))
    (if (= d 10)
        count
        (progn
          (setchar xs count (read-digits d 0))
          (setq count (+ count 1))
          (read-list last)))))

(read-list (readchar))

(loop for i from 1 to (- count 2) do
  (progn
    (if (> i 1) (printchar 32) 0)
    (printnumber (/ (+ (+ (getchar xs (- i 1)) (getchar xs i))
                       (getchar xs (+ i 1)))
                    3))))
(printchar 10)
