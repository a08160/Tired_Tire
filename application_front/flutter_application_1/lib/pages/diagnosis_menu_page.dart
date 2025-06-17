import 'package:flutter/material.dart';
import 'image_picker_modal.dart';
import 'crack_image_modal.dart';

class DiagnosisMenuPage extends StatefulWidget {
  final String userName; // ✅ 추가

  DiagnosisMenuPage({required this.userName});

  @override
  _DiagnosisMenuPageState createState() => _DiagnosisMenuPageState();
}

class _DiagnosisMenuPageState extends State<DiagnosisMenuPage> {
  String? selectedFunction;

  final List<String> diagnosisOptions = ['타이어 공기압 진단', '타이어 균열 진단'];

  void _startDiagnosis() {
    if (selectedFunction == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('기능을 선택해주세요.')));
      return;
    }

    // 기능 분기 처리
    Widget modal;
    if (selectedFunction == '타이어 공기압 진단') {
      modal = ImagePickerModal(userName: widget.userName);
    } else if (selectedFunction == '타이어 균열 진단') {
      modal = CrackImageModal(userName: widget.userName); // ✅ 균열 진단용 새 모달로 분기
    } else {
      modal = Container(); // 혹시 모를 fallback
    }

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => modal,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A171D),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        automaticallyImplyLeading: true, // 뒤로가기 버튼 보이게
        iconTheme: IconThemeData(
          color: Colors.white, // 뒤로가기 버튼 색상
          size: 28, // (선택) 크기 약간 키우기
        ),
        title: null, // 제목 제거
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // ✅ 1. 맨 위 텍스트
              Text(
                '타이어 상태\n사진 한장으로 확인하세요',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 30,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 30),

              // ✅ 이미지 삽입
              Image.asset(
                'assets/tire_check.png',
                width: 336,
                height: 386,
                fit: BoxFit.contain,
              ),
              SizedBox(height: 30),

              DropdownButtonFormField<String>(
                dropdownColor: Colors.white,
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Colors.white,
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 16,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(30),
                    borderSide: BorderSide.none,
                  ),
                ),
                hint: Text('진단 기능을 선택하세요'),
                value: selectedFunction,
                items:
                    diagnosisOptions.map((String value) {
                      return DropdownMenuItem<String>(
                        value: value,
                        child: Text(
                          value,
                          style: TextStyle(color: Colors.black),
                        ),
                      );
                    }).toList(),
                onChanged: (newValue) {
                  setState(() {
                    selectedFunction = newValue;
                  });
                },
              ),
              SizedBox(height: 32),
              ElevatedButton(
                onPressed: () {
                  if (selectedFunction == null) {
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(SnackBar(content: Text('기능을 선택해주세요.')));
                  } else {
                    _startDiagnosis();
                  }
                },
                style: ElevatedButton.styleFrom(
                  minimumSize: Size(double.infinity, 60),
                  backgroundColor:
                      selectedFunction == null ? Colors.grey : Colors.white,
                  foregroundColor:
                      selectedFunction == null ? Colors.white : Colors.black,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                child: Text('진단 시작하기'),
              ),
              // ✅ 3. 안내문 텍스트
              SizedBox(height: 40),
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '🔍 확인해주세요',
                  style: TextStyle(
                    color: Colors.blue.shade300,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              SizedBox(height: 8),
              Text(
                '• 타이어는 안전과 직결되는 부품이기 때문에 보다 보수적으로 판단하여 교체 여부에 대해 의견을 드립니다.\n'
                '• 타이어의 종류, 치수에 따라 예측 정확도의 차이가 있을 수 있습니다.',
                style: TextStyle(color: Colors.white70, fontSize: 14),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
