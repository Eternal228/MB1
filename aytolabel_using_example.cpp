#include <cstdlib>
#include <string>
#include <iostream>

using namespace std;

int main() {
    string python_exe_path = "C:/Projects/Autolabel_test_2/.venv/Scripts/python.exe";   // В этом venv должен быть установлен ultralytics
    string python_module_path = "C:/Projects/Autolabel_test_2/auto_annotating.py";      // Скрипт авторазметки
    string det_model = "C:/Projects/Autolabel_test_2/yolo11x.pt";                       // Модель для разметки выбранных классов
    string sam_model = "C:/Projects/Autolabel_test_2/mobile_sam.pt";                    // Модель для сегментации
    string input = "C:/Projects/Autolabel_test_2/data";                                 // Папка с изображениями
    string output = "C:/Projects/Autolabel_test_2/labeled3";                            // В эту папку будут помещены текстовики с разметкой
    string conf = "0.35";                                                               // Пороговая (минимально допустимая) уверенность
    string command = 
        python_exe_path + " " + 
        python_module_path  + " " + 
        det_model + " " +
        sam_model + " " +
        input + " " + 
        output + " " + 
        conf;

    int ret = system(command.c_str());
    cout << ret;
    return ret;
}
