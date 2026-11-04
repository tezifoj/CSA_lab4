(defun printchar (c)
  (port-out 1 c))

(defun printstring (s)
  (if (= (getchar s 0) 0)
      0
      (progn
        (printchar (getchar s 0))
        (printstring (+ s 1)))))

(defun strlen (s)
  (if (= (getchar s 0) 0)
      0
      (+ 1 (strlen (+ s 1)))))

; печатает цифры числа n <= 0
(defun printdigits (n)
  (if (= n 0)
      0
      (progn
        (printdigits (/ n 10))
        (printchar (- #\0 (% n 10))))))

(defun printnumber (n)
  (if (< n 0)
      (progn
        (printchar #\-)                   ; если число отрицательное, печатаем '-'
        (printdigits n))
      (if (= n 0)
          (printchar #\0)                 ; если 0, печатаем '0'
          (printdigits (- 0 n)))))

; ввод через прерывания
(define in-buf (makestring 64))
(define in-head 0)
(define in-tail 0)

(defint on-input ()
  (progn
    (setchar in-buf (% in-tail 64) (port-in 0))
    (setq in-tail (+ in-tail 1))))

(defun readchar ()
  (if (= in-head in-tail)
      (readchar)
      (let ((c (getchar in-buf (% in-head 64))))
        (setq in-head (+ in-head 1))
        c)))

; читает строку до перевода строки или символа 0 в buf, начиная с позиции i.
; записывает завершающий 0 и возвращает длину строки.
(defun readline (buf i)
  (let ((c (readchar)))
    (if (= c 10)
        (progn (setchar buf i 0) i)
        (if (= c 0)
            (progn (setchar buf i 0) i)
            (progn
              (setchar buf i c)
              (readline buf (+ i 1)))))))

(ei)
