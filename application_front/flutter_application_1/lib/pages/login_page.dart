import 'package:flutter/material.dart';
import 'dart:async';
import 'home_page.dart';

class LoginPage extends StatefulWidget {
  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _phoneController = TextEditingController();
  final _authCodeController = TextEditingController();

  String? _carrier;
  bool _showAuthField = false;
  int _remainingSeconds = 0;
  Timer? _timer;

  final String _expectedAuthCode = "123456";
  final List<String> _carrierOptions = [
    'SKT',
    'KT',
    'LG U+',
    'SKT 알뜰폰',
    'KT 알뜰폰',
    'LG U+ 알뜰폰',
  ];

  void _showCarrierPicker() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        return ListView(
          children:
              _carrierOptions.map((carrier) {
                return ListTile(
                  title: Text(carrier),
                  onTap: () {
                    setState(() {
                      _carrier = carrier;
                    });
                    Navigator.pop(context);
                  },
                );
              }).toList(),
        );
      },
    );
  }

  void _startAuthTimer() {
    setState(() {
      _showAuthField = true;
      _remainingSeconds = 300;
    });
    _timer?.cancel();
    _timer = Timer.periodic(Duration(seconds: 1), (timer) {
      if (_remainingSeconds > 0) {
        setState(() {
          _remainingSeconds--;
        });
      } else {
        timer.cancel();
      }
    });
  }

  void _verifyAndLogin() {
    if (_carrier == null ||
        _phoneController.text.trim().isEmpty ||
        _authCodeController.text.trim().isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('모든 항목을 입력해주세요.')));
      return;
    }
    if (_authCodeController.text.trim() == _expectedAuthCode &&
        _remainingSeconds > 0) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('로그인 성공!')));
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomePage()),
      );
      // TODO: 홈 페이지로 이동 또는 상태 변경
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('인증번호가 올바르지 않거나 시간이 초과되었습니다.')));
    }
  }

  String _formatTime(int seconds) {
    final minutes = seconds ~/ 60;
    final secs = seconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF1A171D),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text('로그인', style: TextStyle(fontWeight: FontWeight.bold)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('통신사', style: TextStyle(color: Colors.white)),
              GestureDetector(
                onTap: _showCarrierPicker,
                child: Container(
                  width: double.infinity,
                  padding: EdgeInsets.symmetric(vertical: 14, horizontal: 16),
                  margin: EdgeInsets.symmetric(vertical: 10),
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(30),
                  ),
                  child: Text(
                    _carrier ?? '통신사를 선택하세요',
                    style: TextStyle(
                      color: _carrier == null ? Colors.white54 : Colors.white,
                    ),
                  ),
                ),
              ),
              SizedBox(height: 16),
              Text('전화번호', style: TextStyle(color: Colors.white)),
              SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _phoneController,
                      style: TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        filled: true,
                        fillColor: Colors.white24,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(30),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 14,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(width: 8),
                  ElevatedButton(
                    onPressed: _startAuthTimer,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      padding: EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 12,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    child: Text('인증번호 발송'),
                  ),
                ],
              ),
              if (_showAuthField) ...[
                SizedBox(height: 10),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _authCodeController,
                        style: TextStyle(color: Colors.white),
                        decoration: InputDecoration(
                          filled: true,
                          fillColor: Colors.white24,
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(30),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 14,
                          ),
                          hintText:
                              _remainingSeconds == 0
                                  ? '제한시간 초과. 인증번호를 재발송해주세요'
                                  : '인증 번호 입력',
                          hintStyle: TextStyle(color: Colors.white54),
                        ),
                      ),
                    ),
                    if (_remainingSeconds > 0) ...[
                      SizedBox(width: 10),
                      Text(
                        _formatTime(_remainingSeconds),
                        style: TextStyle(color: Colors.white),
                      ),
                    ],
                    TextButton(
                      onPressed: _verifyAndLogin,
                      child: Text('확인', style: TextStyle(color: Colors.white)),
                    ),
                  ],
                ),
              ],
              SizedBox(height: 30),
              ElevatedButton(
                onPressed: _verifyAndLogin,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                  minimumSize: Size(double.infinity, 48),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                child: Text('로그인'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _phoneController.dispose();
    _authCodeController.dispose();
    _timer?.cancel();
    super.dispose();
  }
}
