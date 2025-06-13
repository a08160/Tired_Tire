import 'package:flutter/material.dart';
import 'image_picker_page.dart';

class DiagnosisMenuPage extends StatefulWidget {
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

    // 현재는 기능 1, 2 모두 image_picker_page.dart로 이동
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => ImagePickerPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A171D),
      appBar: AppBar(
        title: Text('불량 진단 메뉴', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // ✅ 1. 맨 위 텍스트
            Text(
              '타이어 상태\n사진 한장으로 확인하세요',
              style: TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 30),

            // ✅ 2. 이미지 자리용 공백
            Container(
              width: 200,
              height: 200,
              color: Colors.transparent, // 향후 이미지 삽입 가능
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
                      child: Text(value, style: TextStyle(color: Colors.black)),
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
              onPressed: _startDiagnosis,
              style: ElevatedButton.styleFrom(
                minimumSize: Size(double.infinity, 60),
                backgroundColor: Colors.white,
                foregroundColor: Colors.black,
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
    );
  }
}
