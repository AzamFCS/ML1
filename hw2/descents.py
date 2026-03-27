import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(self,
                 lr_schedule: LearningRateSchedule = TimeDecayLR(),
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                 ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        ...

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        d = self.model.X_train.shape[1]
        self.model.w = np.zeros(d)
        self.model.loss_history = []
        self.model.loss_history.append(self.model.compute_loss())
        self.iteration = 0
        for i in range(self.max_iter):
            delta = self._step()
            current_loss = self.model.compute_loss()
            self.model.loss_history.append(current_loss)

            if np.any(np.isnan(delta)):
                break

            if np.sum(delta ** 2) < self.tolerance:
                break


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - learning_rate * gradient
        return -learning_rate * gradient


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели

        X_train = np.asarray(self.model.X_train)
        y_train = np.asarray(self.model.y_train)
        n = X_train.shape[0]

        indices = np.random.choice(n, size=self.batch_size, replace=False)

        X_batch = X_train[indices]
        y_batch = y_train[indices]

        gradient = self.model.compute_gradients(X_batch, y_batch)
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - learning_rate * gradient
        return -learning_rate * gradient


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)

            for i in range(num_objects):
                g = self.model.compute_gradients(
                    X_train[i:i + 1],
                    y_train[i:i + 1]
                )
                self.grad_memory[i] = g
                self.grad_sum += g

        indices = np.random.choice(num_objects, size=self.batch_size, replace=False)

        for idx in indices:
            g_new = self.model.compute_gradients(X_train[idx:idx + 1], y_train[idx:idx + 1])

            self.grad_sum -= self.grad_memory[idx]
            self.grad_sum += g_new

            self.grad_memory[idx] = g_new

        avg_grad = self.grad_sum / num_objects

        learning_rate = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - learning_rate * avg_grad
        return -learning_rate * avg_grad


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        if (self.velocity is None):
            self.velocity = np.zeros(X_train.shape[1])
        gradient = self.model.compute_gradients(X_train, y_train)
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        self.velocity = self.beta * self.velocity + learning_rate * gradient
        self.model.w = self.model.w - self.velocity
        return -self.velocity


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train

        if (self.m is None and self.v is None):
            self.m = np.zeros(X_train.shape[1])
            self.v = np.zeros(X_train.shape[1])
        gradient = self.model.compute_gradients(X_train, y_train)
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient) ** 2
        m_temp = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_temp = self.v / (1 - self.beta2 ** (self.iteration + 1))

        self.model.w = self.model.w - learning_rate * m_temp / (np.sqrt(v_temp) + self.eps)
        return - learning_rate / (np.sqrt(v_temp) + self.eps) * m_temp


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений
    """

    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решеыния, за него ответственен другой объект
        X, y = self.model.X_train, self.model.y_train
        self.model.w = self.model.loss_function.analytic_solution(X, y)