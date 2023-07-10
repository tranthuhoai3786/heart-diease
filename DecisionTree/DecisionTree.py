# xử lý dữ liệu
import pandas as pd

# tính toán với mảng
import numpy as np

# Tình nhanh giá trị
from collections import Counter


class Node:
    """
    Tạo node cho cây quyết định
    """

    def __init__(
            self,
            Y: list,
            X: pd.DataFrame,
            min_samples_split=None,
            max_depth=None,
            depth=None,
            node_type=None,
            rule=None
    ):
        # Lưu dữ liệu vào node
        self.Y = Y
        self.X = X

        # Lưu tham số
        self.min_samples_split = min_samples_split if min_samples_split else 20 #tách mẫu tối thiểu
        self.max_depth = max_depth if max_depth else 5 #độ sâu tối đa

        # Độ sâu mặc định hiện tại của node là 0
        self.depth = depth if depth else 0

        # Trích chọn đặc trưng thành list
        self.features = list(self.X.columns)

        # Loại node, nếu chưa thuộc loại nàu là là node gốc
        self.node_type = node_type if node_type else 'root'

        # Quy tắc tách
        self.rule = rule if rule else ""

        # Tính số lượng Y trong node
        self.counts = Counter(Y)

        # Lấy GINI dựa trên phân phối Y
        self.gini_impurity = self.get_GINI()

        # Sắp xếp số lượng và lưu dự đoán cuối cùng của node
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Lấy item cuối cùng
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # Lưu vào object attribute. Node này sẽ dự đoán lớp
        self.yhat = yhat

        # Lưu số lượng quan sát trong nút
        self.n = len(Y)

        # Khởi tạo node trái phải là node rỗng
        self.left = None
        self.right = None

        # Giá trị mặc định cho các phần tách (đặc trưng tốt nhất, giá trị tốt nhất)
        self.best_feature = None
        self.best_value = None

    @staticmethod
    def GINI_impurity(y1_count: int, y2_count: int) -> float:
        """
        tính  GINI = 1 - tong (pi ^2)
        """
        # Đảm bảo đúng type
        if y1_count is None:
            y1_count = 0

        if y2_count is None:
            y2_count = 0

        # Lấy tổng quan sát
        n = y1_count + y2_count

        # Nếu n = 0 trả về GINI thấp nhất có thể
        if n == 0:
            return 0.0

        # Lấy xác suất từng lớp
        p1 = y1_count / n
        p2 = y2_count / n

        # Tính GINI
        gini = 1 - (p1 ** 2 + p2 ** 2)

        # Trả về GINI
        return gini

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Tính trung bình của danh sách đã cho
        """
        return np.convolve(x, np.ones(window), 'valid') / window

    def get_GINI(self):
        """
        Hàm tính GINI của node
        """
        # Lấy số lượng 0 và 1
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)

        # Lấy GINI
        return self.GINI_impurity(y1_count, y2_count)

    def best_split(self) -> tuple:
        """
        Với các đặc trưng X và target Y tính toán mức phân chia tốt nhất cho cây quyết định
        """
        # Tạo tập dữu liệu để tách
        df = self.X.copy()
        df['Y'] = self.Y

        # Lấy GINI cho đầu vào cơ sở
        GINI_base = self.get_GINI()

        # Tìm cách phân chia nào mang lại mức tăng GINI tốt nhất
        max_gain = 0

        # Đặc trưng tốt nhất mặc định và split tốt nhất
        best_feature = None
        best_value = None

        for feature in self.features:
            # Xóa các giá trị thiếu
            Xdf = df.dropna().sort_values(feature)

            # Sắp xếp các giá trị và lấy giá trị trung bình
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # chia tập dữ liệu
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                # Lấy phân phối Y
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0,0), right_counts.get(1, 0)

                # Lấy GINI trái và phải
                gini_left = self.GINI_impurity(y0_left, y1_left)
                gini_right = self.GINI_impurity(y0_right, y1_right)

                # Lấy số lượng quan sát từ split dữ liệu trái và phải
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                # Tính trọng số cho mỗi node
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Tính toán GINI có trọng số
                wGINI = w_left * gini_left + w_right * gini_right

                # Tính mức tăng của GINI
                GINIgain = GINI_base - wGINI

                # Kiểm tra xem đây có phải split tốt nhất k?
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value

                    # Mức tăng tốt nhất hiện tại
                    max_gain = GINIgain

        return (best_feature, best_value) # trả về đặc trưng tốt nhất và giá trị tốt nhất

    def grow_tree(self):
        """
        Tạo cây quyết định
        """
        # Tạo dataframe từ data
        df = self.X.copy()
        df['Y'] = self.Y

        # Nếu độ sâu chưa phải lớn nhất và phân chia mẫu chưa phải nhỏ nhất thì phân chia nhỏ hơn
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Phân chia tốt nhất
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Lưu split tốt nhất vào nút hiện tại
                self.best_feature = best_feature
                self.best_value = best_value

                # Lấy các node trái và phải
                left_df, right_df = df[df[best_feature] <= best_value].copy(), df[df[best_feature] > best_value].copy()

                # Tạo node trái và phải
                left = Node(
                    left_df['Y'].values.tolist(),
                    left_df[self.features],
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                )

                self.left = left
                self.left.grow_tree()

                right = Node(
                    right_df['Y'].values.tolist(),
                    right_df[self.features],
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=4):
        """
        Print thông tin của cây quyết định
        """
        # Xác định số lượng không gian
        const = int(self.depth * width ** 1.5)-4
        spaces = "-" * const

        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
            print(f"{' ' * const}   | GINI cua node: {round(self.gini_impurity, 2)}")
            print(f"{' ' * const}   | Phan phoi lop trong node: {dict(self.counts)}")
            print(f"{' ' * const}   | Lop du doan: {self.yhat}")

    def print_tree(self):
        """
        Print cây quyết định từ node hiện tại đến node cuối
        """
        self.print_info()

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()

    def predict(self, X: pd.DataFrame):
        """
        Dự đoán
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})

            predictions.append(self.predict_obs(values))#dự đoán cho từng hàng trong dataset

        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Dự đoán class từ 1 tập đặc trưng
        """
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Duyệt qua các nút cho đến cuối
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            #số lượng mẫu trong nút hiện tại (n) có nhỏ hơn ngưỡng tối thiểu để tiếp tục phân chia
            if cur_node.n < cur_node.min_samples_split:
                break

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right

        return cur_node.yhat#dự đoán

